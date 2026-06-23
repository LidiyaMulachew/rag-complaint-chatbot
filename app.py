import gradio as gr

# =====================================================
# RAG PIPELINE FUNCTION
# =====================================================
def rag_pipeline(question):
    # Step 1: Retrieve relevant chunks
    results = retrieve_chunks(question, k=5)

    # Step 2: Build context
    context = "\n\n".join(results["documents"][0])

    # Step 3: Build prompt
    prompt = build_prompt(context, question)

    # Step 4: LLM response
    response = llm(prompt)

    # Step 5: Clean answer
    answer = response[0]["generated_text"].replace(prompt, "").strip()

    # Step 6: Get sources
    sources = results["documents"][0]

    return answer, sources


# =====================================================
# CHAT FUNCTION (for Gradio ChatInterface)
# =====================================================
def chat(message, history):
    answer, sources = rag_pipeline(message)

    # format sources nicely
    formatted_sources = "\n\n".join(
        [f"Source {i+1}:\n{src}" for i, src in enumerate(sources[:3])]
    )

    return f"{answer}\n\n---\nSources:\n{formatted_sources}"


# =====================================================
# GRADIO UI (UPGRADED)
# =====================================================
demo = gr.ChatInterface(
    fn=chat,
    title="CrediTrust Financial RAG Chatbot",
    description="Ask questions about customer complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers."
)


# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    demo.launch()