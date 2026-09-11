import gradio as gr
from integrated_checker import MedicalFactChecker

engine = MedicalFactChecker()

def research_pipeline(hinglish_claim):
    # Stage 1: The Llama Translation
    english_claim = engine.translate_hinglish(hinglish_claim)
    
    # Stage 2: The PubMed Retrieval
    matches, scores = engine.retrieve_evidence(english_claim, k=1)
    evidence = matches[0]['explanation']
    db_claim = matches[0]['claim']
    
    # Stage 3: The PubMedBERT Verdict
    verdict = engine.get_verdict(english_claim, evidence)
    
    return english_claim, db_claim, evidence, verdict

# Building the "Show-Worthy" Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ ShieldHealth: Medical Fact-Checking Pipeline")
    gr.Markdown("### Research Ablation: Hinglish -> Translation -> Retrieval -> NLI")
    
    with gr.Row():
        user_input = gr.Textbox(label="Enter Hinglish Health Claim", placeholder="e.g., Coronavirus vaccine se cancer hota hai?")
        submit_btn = gr.Button("Analyze", variant="primary")
    
    with gr.Accordion("Step 1: LLM Translation (Llama-3.2)", open=True):
        output_eng = gr.Textbox(label="Translated English Reference")
        
    with gr.Accordion("Step 2: Semantic Retrieval (FAISS + PubMed)", open=True):
        output_match = gr.Textbox(label="Top Database Match")
        output_evidence = gr.TextArea(label="Medical Evidence Found")
        
    with gr.Column():
        gr.Markdown("### Final Decision")
        output_verdict = gr.Label(label="PubMedBERT Verdict")

    submit_btn.click(
        fn=research_pipeline, 
        inputs=user_input, 
        outputs=[output_eng, output_match, output_evidence, output_verdict]
    )

if __name__ == "__main__":
    demo.launch()