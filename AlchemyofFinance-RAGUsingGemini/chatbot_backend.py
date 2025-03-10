from sentence_transformers import util, SentenceTransformer

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available 
import torch
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm # for progress bars

import fitz
import textwrap


class Model:

    def __init__(self):

        # self.model = None
        # self.tokenizer = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self.instantiate_model_and_tokenizer("./soros_llm_model")
        self.embeddings, self.pages_and_chunks = self.set_embeddings()
        self.embedding_model = self.set_embedding_model()

    def print_wrapped(self, text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        print(wrapped_text)

    def instantiate_model_and_tokenizer(self, model_path):

        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        with tqdm(total=2, desc="Loading model and tokenizer") as pbar:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )
            pbar.update(1)
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            pbar.update(1)

        # self.model = model
        # self.tokenizer = tokenizer
        return model, tokenizer


    def sample_response(self):
        input_text = "Hello, how are you?"
        inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")  # Move inputs to CUDA

        outputs = self.model.generate(**inputs)
        print(self.tokenizer.decode(outputs[0]))

    
    def get_model_num_params(self):
        return sum([param.numel() for param in self.model.parameters()])
    
    def get_model_mem_size(self):
        """
        Get how much memory a PyTorch model takes up.

        See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
        """
        # Get model parameters and buffer sizes
        mem_params = sum([param.nelement() * param.element_size() for param in self.model.parameters()])
        mem_buffers = sum([buf.nelement() * buf.element_size() for buf in self.model.buffers()])

        # Calculate various model sizes
        model_mem_bytes = mem_params + mem_buffers # in bytes
        model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
        model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

        return {"model_mem_bytes": model_mem_bytes,
                "model_mem_mb": round(model_mem_mb, 2),
                "model_mem_gb": round(model_mem_gb, 2)}
    

    def prompt_chat_template(self, input_text):        
        print(f"Input text:\n{input_text}")

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
            "content": input_text}
        ]

        # Apply the chat template
        self.prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                            tokenize=False, # keep as raw text (not tokenized)
                                            add_generation_prompt=True)
        
        print(f"\nPrompt (formatted):\n{self.prompt}")



    def get_model_response(self, input_text):
        
        self.prompt_chat_template(input_text)

        # Tokenize the input text (turn it into numbers) and send it to GPU
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").to("cuda")
        print(f"Model input (tokenized):\n{input_ids}\n")

        # Generate outputs passed on the tokenized input
        # See generate docs: https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/text_generation#transformers.GenerationConfig 
        outputs = self.model.generate(**input_ids,
                                    max_new_tokens=256) # define the maximum number of new tokens to create
        
        print(f"Model output (tokens):\n{outputs[0]}\n")

        # Decode the output tokens to text
        outputs_decoded = self.tokenizer.decode(outputs[0])
        print(f"Model output (decoded):\n{outputs_decoded}\n")

        outputs_decoded = outputs_decoded.replace(self.prompt, '').replace('<bos>', '').replace('<eos>', '')

        return outputs_decoded


    # Read the embedding csv file and return the embedding matrix 
    # The embeddings are stored in a csv file named text_chunks_and_embeddings_soros_df.csv
    def set_embeddings(self):        

        # Import texts and embedding df
        text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_soros_df.csv")

        # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

        # Convert texts and embedding df to list of dicts
        pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

        # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
        embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(self.device)
        print("embeddings.shape : " , embeddings.shape)

        return embeddings, pages_and_chunks
    
    def set_embedding_model(self):
        embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device=self.device) # choose the device to load the model to
        return embedding_model
    

    def retrieve_relevant_resources(self,
                                query: str,
                                embeddings: torch.tensor,                                
                                n_resources_to_return: int=5,
                                print_time: bool=True):
        """
        Embeds a query with model and returns top k scores and indices from embeddings.
        """

        # model: SentenceTransformer=self.embedding_model

        # Embed the query
        query_embedding = self.embedding_model.encode(query, 
                                    convert_to_tensor=True) 

        # Get dot product scores on embeddings
        # start_time = timer()
        dot_scores = util.dot_score(query_embedding, embeddings)[0]
        # end_time = timer()

        # if print_time:
        #     print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

        scores, indices = torch.topk(input=dot_scores, 
                                    k=n_resources_to_return)

        return scores, indices
    
    def print_top_results_and_scores(self,
                                    query: str,                                                                       
                                    n_resources_to_return: int=5):
        """
        Takes a query, retrieves most relevant resources and prints them out in descending order.

        Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
        """
        
        # pages_and_chunks: list[dict]
        pages_and_chunks = self.pages_and_chunks

        # embeddings: torch.tensor
        embeddings = self.embeddings

        scores, indices = self.retrieve_relevant_resources(query=query,
                                                    embeddings=embeddings,
                                                    n_resources_to_return=n_resources_to_return)
        
        print(f"Query: {query}\n")
        print("Results:")

        print("Score type : ", type(scores))
        print("Indices type : ", type(indices))

        top_result_complete = ""
        # Loop through zipped together scores and indicies
        for score, index in zip(scores, indices):
            print(f"Score: {score:.4f}")
            # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
            self.print_wrapped(pages_and_chunks[index]["sentence_chunk"])
            top_result_complete += pages_and_chunks[index]["sentence_chunk"] + "\n"

            # Print the page number too so we can reference the textbook further and check the results
            print(f"Page number: {pages_and_chunks[index]['page_number']}")
            print("\n")
            top_result_complete += f"Page number: {pages_and_chunks[index]['page_number']}" + "\n"
        
        return top_result_complete
    
    def prompt_formatter(self,
                        query: str, 
                        context_items: list[dict]) -> str:
        """
        Augments query with text-based context from context_items.
        """
        # Join context items into one dotted paragraph
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

        ## Create a base prompt with examples to help the model
        ## Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
        ## We could also write this in a txt file and import it in if we wanted.


        base_prompt = """Based on the following context items, please answer the query.
                Respond the greetings with a greeting.
                Also rememeber to address the person by their name if the name is mentioned.
                Give yourself room to think by extracting relevant passages from the context before answering the query.
                Don't return the thinking, only return the answer.
                Make sure your answers are as explanatory as possible.
                Use the following examples as reference for the ideal answer style.
                \nExample 1:
                Query: What is reflexivity theory?
                Answer: Reflexivity is the feedback loop between perception and reality. In financial markets, the actions of participants, influenced by their perceptions, can alter the fundamentals of the market they are observing.The Key Idea is markets do not merely reflect reality; they can shape it. Investors' decisions, based on their beliefs, create market conditions that validate or invalidate those beliefs.
                \nExample 2:
                Query: What is participating bias?
                Answer: Participating bias occurs when market participants do not merely observe or react to market conditions but actively influence them through their decisions and actions. This bias is the result of participants’ beliefs, perceptions, and actions becoming part of the market's reality, creating a feedback loop.
                \nExample 3:
                Query: Define international debt problem?
                Answer: The international debt problem as part of his broader analysis of global financial systems and the interplay between creditors and debtors on a global scale. The international debt problem refers to the challenges arising when countries, particularly developing nations, accumulate unsustainable levels of debt, often leading to economic crises, default risks, and systemic instability.
                \nNow use the following context items to answer the user query:
                {context}
                \nRelevant passages: <extract relevant passages from the context here>
                User query: {query}
                Answer:"""

        # Update base prompt with context items and query   
        base_prompt = base_prompt.format(context=context, query=query)

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
            "content": base_prompt}
        ]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                            tokenize=False,
                                            add_generation_prompt=True)
        return prompt
    

    def ask(self,
        query, 
        temperature=0.7,
        max_new_tokens=512,
        format_answer_text=True, 
        return_answer_only=True):
        """
        Takes a query, finds relevant resources/context and generates an answer to the query based on the relevant resources.
        """
        
        # Get just the scores and indices of top related results
        scores, indices = self.retrieve_relevant_resources(query=query,
                                                    embeddings=self.embeddings)
        
        # Create a list of context items
        context_items = [self.pages_and_chunks[i] for i in indices]

        # Add score to context item
        for i, item in enumerate(context_items):
            item["score"] = scores[i].cpu() # return score back to CPU 
            
        # Format the prompt with context items
        prompt = self.prompt_formatter(query=query,
                                context_items=context_items)
        
        # Tokenize the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate an output of tokens
        outputs = self.model.generate(**input_ids,
                                    temperature=temperature,
                                    do_sample=True,
                                    max_new_tokens=max_new_tokens)
        
        # Turn the output tokens into text
        output_text = self.tokenizer.decode(outputs[0])

        if format_answer_text:
            # Replace special tokens and unnecessary help message
            output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

        # Only return the answer without the context items
        if return_answer_only:
            return output_text
        
        return output_text, context_items



if __name__ == "__main__":

    # model = Model()
    # loaded_model, loaded_tokenizer = model.instantiate_model_and_tokenizer("./soros_llm_model")
    print( "Inside the main function")
    # print(loaded_model)

