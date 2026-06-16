from streamlit_app.services.rag_service import (
    get_vector_pipeline
)

pipe = get_vector_pipeline()

print("Loaded successfully")