from src.app import create_gradio_demo

if __name__ == "__main__":
    demo = create_gradio_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
