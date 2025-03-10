from flask import Flask, jsonify, request
from chatbot_backend import Model

class SorosBackendFlaskApp(Flask):
    def __init__(self, import_name, **kwargs):
        super().__init__(import_name, **kwargs)
        self.register_routes()
        self.instantiate_model()
    
    def instantiate_model(self):
        print('Inside Instantiate Model')
        self.model = Model()
        self.model.sample_response()

    def register_routes(self):

        """Register routes for the application."""
        @self.route('/')
        def home():
            return jsonify({"message": "Welcome to the George Soros Bot Flask App!"})

        @self.route('/echo', methods=['POST'])
        def echo():
            data = request.json
            return jsonify({"you_sent": data})

        @self.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "OK"})
        
        @self.route('/test_response', methods=['GET'])
        def test_response():
            input_text = "explain about the reflexivity theory"
            model_response = self.model.get_model_response("explain about the reflexivity theory")
            print(model_response)
            return jsonify({"model_response": model_response})

        @self.route('/response', methods=['GET'])
        def response():
            prompt = request.args.get('prompt', default="Thanks to all your responses", type=str)
            model_response = self.model.get_model_response(prompt)
            print(model_response)
            # self.model.print_top_results_and_scores(query="")
            return jsonify({"response": model_response})
        
        @self.route('/test_relavant_response', methods=['GET'])
        def test_relavant_response():
            test_relevane_response = self.model.print_top_results_and_scores(query="real estate short")
            return jsonify({"response": test_relevane_response})
        
        @self.route('/test_ask_response', methods=['GET'])
        def test_ask_response():
            answer, context_items = self.model.ask(query="what is conglomerate boom", 
                                                temperature=0.7,
                                                max_new_tokens=512,
                                                return_answer_only=False)
            context_data_from_book = ""
            for context_item in context_items:
                relevant_sentence_chunk = context_item['sentence_chunk']
                relevant_page_number = context_item['page_number']
                context_data_from_book += f"Page {relevant_page_number}: {relevant_sentence_chunk}\n"

            return jsonify({"response": answer,
                            "context_data_from_book": context_data_from_book})
        
        @self.route('/ask_response', methods=['GET'])
        def ask_response():
            prompt = request.args.get('prompt', default="What is a prompt ?", type=str)
            answer, context_items = self.model.ask(query=prompt, 
                                                temperature=0.7,
                                                max_new_tokens=512,
                                                return_answer_only=False)
            context_data_from_book = ""
            for context_item in context_items:
                relevant_sentence_chunk = context_item['sentence_chunk']
                relevant_page_number = context_item['page_number']
                context_data_from_book += f"Page {relevant_page_number}: {relevant_sentence_chunk}\n"

            return jsonify({"response": answer,
                            "context_data_from_book": context_data_from_book})


    def run_server(self, host='127.0.0.1', port=5000, debug=False):
        """Run the Flask server."""
        self.run(host=host, port=port, debug=debug)


# Example of using the CustomFlaskApp
if __name__ == '__main__':
    app = SorosBackendFlaskApp(__name__)
    app.run_server(debug=True)
