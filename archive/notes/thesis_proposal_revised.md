
4 OBJETIVOS

4.1 Geral

Desenvolver e validar modelos multimodais de inteligência artificial para a predição do risco de incidência e progressão da osteoartrite de joelhos (OAJ), integrando dados clínicos-epidemiológicos longitudinais e características extraídas de imagens radiográficas por meio de aprendizado profundo, bem como simular visualmente a degeneração articular futura.

4.2 Específicos

1.  **Definir e otimizar o componente clínico do modelo multimodal:** Comparar sistematicamente o desempenho de modelos clínicos simples (Regressão Logística Stepwise) versus algoritmos complexos (XGBoost, Random Forest, Redes Neurais) na identificação de OAJ prevalente, visando selecionar o vetor de características clínicas mais eficiente e parcimonioso para a fusão com dados de imagem.
2.  **Desenvolver o componente de imagem:** Validar e aprimorar modelos de Redes Neurais Convolucionais (CNNs) para a extração automática de biomarcadores radiográficos a partir das imagens do ELSA-Brasil MSK.
3.  **Construir o modelo preditivo multimodal:** Integrar os componentes clínico (selecionado na etapa 1) e de imagem (etapa 2) em uma arquitetura de fusão para prever a incidência e progressão da doença em seguimento longitudinal.
4.  **Validar externamente:** Avaliar a acurácia e a capacidade de generalização dos modelos preditivos em coortes internacionais.
5.  **Simulação Generativa:** Demonstrar a evolução radiográfica da doença por meio da criação de imagens sintéticas baseadas em inteligência artificial generativa.

4.3 Hipóteses

1.  A predição de desfechos longitudinais (incidência/progressão) requer uma abordagem multimodal, onde a combinação de dados clínicos e de imagem supera o desempenho de qualquer modalidade isolada.
2.  Para a caracterização clínica transversal (prevalência), modelos lineares simples são suficientes e preferíveis a modelos "black-box" complexos, servindo como input clínico otimizado para os modelos longitudinais mais robustos.
3.  É possível representar visualmente a degeneração articular futura prevista pelos modelos por meio de técnicas de inteligência artificial generativa.

5 MÉTODOS

5.1 Desenho e amostra

O estudo utiliza dados longitudinais do ELSA-Brasil Musculoesquelético (ELSA-Brasil MSK), compreendendo a linha de base (2012-2014) e o seguimento (2016-2018).
Na linha de base, foram avaliados 2901 participantes (5660 joelhos com radiografias). No seguimento, 2523 participantes retornaram para novas radiografias, permitindo a definição dos desfechos de incidência e progressão.

5.2 Aspectos éticos

O estudo respeita todos os preceitos éticos (Resolução 466/12), com aprovação pelos Comitês de Ética em Pesquisa (CEP) das instituições envolvidas e da CONEP (CAAE 0186.1.203.000-06 e complementares).

5.3 e 5.4 Dados e Imagem

(Mantidos conforme versão anterior: Coleta de dados clínicos padronizados e aquisição radiográfica com protocolo específico e controle de qualidade rigoroso).

5.5 Estratégia de Modelagem Computacional

A estratégia do doutorado divide-se em duas fases interconectadas: (1) Otimização dos Inputs (Clínico e Imagem) utilizando dados transversais, e (2) Fusão Multimodal para Predição Longitudinal.

5.5.1 Fase 1: Otimização do Componente Clínico (Estudo Transversal)

O objetivo desta fase foi definir a melhor representação dos dados clínicos para alimentar o modelo final, evitando a inserção de ruído ou complexidade desnecessária.
Para isso, realizou-se um estudo comparativo extensivo utilizando o desfecho de OAJ radiográfica prevalente. Foram confrontados modelos lineares clássicos (Regressão Logística Stepwise) contra algoritmos de estado da arte em aprendizado de máquina (XGBoost, Random Forest, Redes Neurais MLP).
Os modelos foram testados em cenários de "Triagem" (apenas fatores de risco demográficos/históricos) e "Case Finding" (com adição de sintomas).
**Resultado da Decisão Metodológica:** A análise demonstrou que modelos lineares simples (Stepwise com 5 variáveis: idade, IMC, cirurgia prévia, trauma e sintomas) apresentam desempenho discriminativo (AUC > 0.81) estatisticamente equivalente aos modelos complexos de "caixa preta". Portanto, o **modelo logístico simples foi selecionado** para compor o vetor de entrada clínico do modelo multimodal, garantindo parcimônia e interpretabilidade sem perda de acurácia.

5.5.2 Fase 2: Modelo de Predição Multimodal (Estudo Longitudinal)

O eixo central da tese é a construção de um modelo preditivo para **incidência e progressão** da OAJ. Este modelo integrará:
1.  **Input Clínico:** As variáveis selecionadas e ponderadas na Fase 1 (Score de Risco Clínico ou o vetor de 5 variáveis).
2.  **Input de Imagem:** Vetores de características (feature embeddings) extraídos das radiografias basais por uma Rede Neural Convolucional (DenseNet-161), previamente treinada e validada na coorte (publicação anterior).

**Arquiteturas de Fusão:**
Serão testadas estratégias de:
*   **Fusão Tardia (Late Fusion):** Combinação das probabilidades de risco geradas independentemente pelos modelos clínico e de imagem.
*   **Fusão Intermediária (Feature Fusion):** Concatenação do vetor de características clínicas com os embeddings profundos da imagem antes das camadas de classificação final, permitindo que o modelo aprenda interações não-lineares entre fenótipos clínicos e padrões radiográficos sutis.
Algoritmos de Gradient Boosting (XGBoost/LightGBM) e Redes Neurais Multimodais serão utilizados para processar essa fusão e estimar o risco futuro de doença.

5.5.3 Modelos Generativos (Simulação Visual)

Paralelamente à predição numérica de risco, modelos generativos (GANs/Diffusion Models) serão treinados para traduzir o prognóstico em uma imagem sintética, visualizando a aparência esperada do joelho no futuro (ex: aparecimento de osteófitos ou redução do espaço articular) com base nos dados basais.

6 RESULTADOS PARCIAIS

A tese já apresenta resultados consolidados que fundamentam as etapas subsequentes:

6.1 Definição do Componente Clínico

Foi concluído o estudo comparativo para definição do braço clínico do modelo. Resultados principais:
1.  **Simplicidade vs. Complexidade:** Para dados clínicos estruturados, algoritmos complexos (ML) não superam modelos lineares bem ajustados. Isso simplifica a arquitetura do modelo multimodal final.
2.  **Validação de Variáveis:** Confirmou-se a robustez de 5 preditores-chave (Idade, IMC, Trauma, Cirurgia, Sintomas) como suficientes para capturar o risco clínico basal.
3.  **Inutilidade de Biomarcadores Complexos:** A bioimpedância e antropometria avançada não agregaram valor preditivo, permitindo sua exclusão do modelo final e facilitando a aplicabilidade externa.

Estes achados (em fase de submissão) balizam a construção do modelo longitudinal, que focará a complexidade computacional no processamento de imagem e na fusão multimodal, mantendo o input clínico eficiente e transparente.

6.2 Validação do Componente de Imagem

(Citar brevemente sucesso prévio da DenseNet-161/RNC na identificação de casos prevalentes, que servirá de backbone para a extração de features longitudinais).

6.3 Contribuições Técnicas (Leitura e Curadoria)

(Manter texto sobre a leitura longitudinal das 5660 radiografias e calibração com centros internacionais, garantindo o "Ground Truth" para o treinamento dos modelos longitudinais).
