import json
from transformers import pipeline

# Carregar contexto
with open("sample_context.json", "r", encoding="utf-8") as f:
    context_data = json.load(f)

# Transformar o contexto em string legível
context = json.dumps(context_data, ensure_ascii=False, indent=2)

# Inicializar pipeline com modelo maior
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def responder(pergunta: str) -> str:
    # Prompt mais inteligente
    prompt = (
        "Você é um atendente da Livraria do Porto. "
        "Responda APENAS com base nas informações fornecidas. "
        "Se a pergunta não puder ser respondida pelo contexto, diga claramente: "
        "'Não tenho essa informação.'\n\n"
        f"Informações disponíveis:\n{context}\n\n"
        f"Pergunta: {pergunta}\nResposta:"
    )

    resposta = qa_pipeline(prompt, max_length=200, clean_up_tokenization_spaces=True)
    return resposta[0]["generated_text"].strip()

# -------------------------
# Loop interativo (3 perguntas)
# -------------------------
if __name__ == "__main__":
    interacoes = []

    print("🤖 Bem-vindo à Livraria do Porto!")
    print("Você pode fazer até 3 perguntas.\n")

    for i in range(3):
        pergunta = input(f"Pergunta {i+1}: ")
        resposta = responder(pergunta)

        print(f"\nResposta: {resposta}\n")

        interacoes.append({"pergunta": pergunta, "resposta": resposta})

    # Gerar resumo final
    print("\n--- Resumo das interações ---")
    resumo = ""
    for i, interacao in enumerate(interacoes, 1):
        resumo += f"{i}. Pergunta: {interacao['pergunta']}\n"
        resumo += f"   Resposta: {interacao['resposta']}\n"

    print(resumo)

    # Salvar em arquivo
    with open("resumo_interacao.txt", "w", encoding="utf-8") as f:
        f.write(resumo)

    print("📄 Resumo salvo em resumo_interacao.txt")
