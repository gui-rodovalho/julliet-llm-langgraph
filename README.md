Juliett - Assitente Virtual - Inteligência Artificial Generativa
Descrição
Juliett é uma aplicação web feita utilizando Streamlit e Langgraph em Python, que cria uma inteligência artificial generativa, baseada em modelos de linguagem natural (LLM), implementando um chatboot e uma assistente virtual. Como forma de torná-la apta a responder questões muito específcas e a criar documentos complexos a Juliett utiliza a técnica denominada RAG (Retrieval-Augmented Generation) que é uma técnica que combina geração de texto baseada em modelos de linguagem com a recuperação de informações em fontes externas para complementar seu conhecimento. Os dados são fornecidos por meio de arquivos no formato .pdf.

Funcionalidades
Chatboot pra perguntas e respostas sobre conhecimentos específicos fornecidos por meio do RAG;
RAG por meio de arquivos no formato .pdf;
Interface web simples, intuitiva e poderosa.
Requisitos mínimos
Utilizando LLM local - modelo de 8b de parâmetros (ex. Llama 3.1 8b):
Utilizando chip apple (M1, M2 ou M3)- mínimo 18gb de memória RAM;
Utilizando GPU - mínimo GPU com 8gb de memória VRAM;
Utilizando LLM em nuvem:
Mínimo 8gb de memória RAM.
Instalação
Pré-Requisitos
Ter o Docker instalado na sua máquina.
Passos para a instalação
1. Obter o Dockerfile + arquivos de configuração:
Obter a pasta compactada contendo dockerfile e outros arquivos necessários para a instalação.

2. Construa a imagem Docker
Abra o terminal dentro da pasta do projeto e execute o seguinte comando para construir a imagem Docker: docker build -t juliett .

3. Rodar a imgem criada
Utilize o comando docker run -p 8501:8501 juliett

Limitações
Atualmente a Juliett fz o RAG apenas a partir de documentos no formato .pdf;
A utilização da IA rodando modelos de LLM local depende do poder de processamento da máquina onde será instalada, mesmo seguindo os reuisitos mínimos pode ser experimentado uma demora na produção de respostas, respostas pouco elaboradas ou incorretas.
Autores
Guilherme Felipe Breetz Rodovalho - Desenvolvedor principal - GitHub
