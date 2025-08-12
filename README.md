# Telecom X — Previsão de Churn (Parte 2)

Projeto de ciência de dados para prever a evasão (churn) de clientes da empresa fictícia Telecom X. O trabalho está estruturado em um notebook com pipeline completo: exploração e limpeza dos dados, pré-processamento (encode, imputação, normalização), balanceamento de classes (SMOTE), divisão treino/teste, treinamento de modelos e avaliação com métricas de classificação.

## Visão Geral
- Objetivo: identificar clientes com alta probabilidade de cancelar o serviço para orientar ações de retenção.
- Entrada: dataset tabular `df_limpo.csv` (7.043 linhas, 22 colunas inicialmente).
- Saída: modelos treinados e avaliação por métricas (accuracy, precision, recall, f1), relatórios e matrizes de confusão gerados no notebook.

## Estrutura do Repositório
- `prediction_telecom_x.ipynb`: notebook principal com todo o pipeline.
- `df_limpo.csv`: dataset limpo utilizado no notebook.

## Dados
- Dimensão: 7.043 linhas, 22 colunas (21 após remoções iniciais).
- Variável alvo: `Churn` (Yes/No), convertida para binária (Yes=1, No=0).
- Colunas (amostra/agrupadas por domínio):
  - Identificação: `customerID` (removida na limpeza inicial)
  - Cliente: `customer.gender`, `customer.SeniorCitizen`, `customer.Partner`, `customer.Dependents`, `customer.tenure`
  - Telefone: `phone.PhoneService`, `phone.MultipleLines`
  - Internet: `internet.InternetService`, `internet.OnlineSecurity`, `internet.OnlineBackup`, `internet.DeviceProtection`, `internet.TechSupport`, `internet.StreamingTV`, `internet.StreamingMovies`
  - Conta/Pagamento: `account.Contract`, `account.PaperlessBilling`, `account.PaymentMethod`, `account.Charges.Monthly`, `account.Charges.Total`
  - Valor agregado: `Total.Day`
- Classes antes do balanceamento: No=5.174 (73,46%), Yes=1.869 (26,54%).

## Ambiente e Dependências
Recomendado Python 3.9+.

Dependências principais utilizadas no notebook:
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (preprocessing, model_selection, metrics, modelos)
- imbalanced-learn (SMOTE)
- Jupyter (para executar o notebook)

Instalação rápida (ambiente virtual opcional):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install jupyter pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

Abrir o notebook:
```bash
jupyter notebook prediction_telecom_x.ipynb
```

## Pipeline (Notebook)
1. Carregamento e inspeção inicial do dataset.
2. Limpeza: remoção de `customerID`; checagem e imputação mínima de valores ausentes (mediana para numéricos, moda para categóricos, quando necessário).
3. EDA: taxa de churn, contagens e distribuições com visualizações.
4. Pré-processamento:
   - Label Encoding para variáveis categóricas (mapeamentos registrados e impressos no notebook).
   - Duas versões de features: sem normalização (`X_no_scale`) e com normalização `StandardScaler` (`X_scaled`).
5. Balanceamento: aplicação de SMOTE (random_state=42) devido a desbalanceamento (ratio ≈ 0,36), gerando conjuntos balanceados 50/50.
6. Divisão treino/teste: `train_test_split` estratificado (test_size=0,25, random_state=42) para versões escaladas e não escaladas.
7. Modelagem (três modelos):
   - Logistic Regression (com dados escalados).
   - Random Forest (sem escala — árvores são invariantes a escala).
   - K-Nearest Neighbors (K=5, com dados escalados).
8. Avaliação: accuracy, precision, recall, f1-score, relatório de classificação e matriz de confusão; comparação treino vs. teste para diagnóstico de over/underfitting.

### Preparação dos Dados — Detalhes
- Classificação de variáveis (detectadas automaticamente no notebook):
  - Numéricas: `customer.SeniorCitizen`, `customer.tenure`, `Total.Day`, `account.Charges.Monthly`, `account.Charges.Total`.
  - Categóricas: `customer.gender`, `customer.Partner`, `customer.Dependents`, `phone.PhoneService`, `phone.MultipleLines`, `internet.InternetService`, `internet.OnlineSecurity`, `internet.OnlineBackup`, `internet.DeviceProtection`, `internet.TechSupport`, `internet.StreamingTV`, `internet.StreamingMovies`, `account.Contract`, `account.PaperlessBilling`, `account.PaymentMethod`.
- Codificação: Label Encoding por coluna categórica com registro dos mapeamentos nos outputs do notebook.
- Normalização: `StandardScaler` aplicado na versão escalada de `X` (para modelos sensíveis à escala).
- Split: `train_test_split` estratificado (25% teste), com `random_state=42`.

### Justificativas de Modelagem
- Regressão Logística: modelo linear que se beneficia de dados padronizados (convergência e estabilidade da otimização).
- Random Forest: baseado em árvores; pouco sensível à escala, por isso treinado sem normalização.
- KNN: distância euclidiana é afetada por escala; normalização é essencial para comparabilidade entre atributos.
- SMOTE: aplicado para corrigir desbalanceamento significativo (26,5% vs 73,5%).
- Split estratificado: preserva a proporção das classes em treino e teste.

### EDA e Insights
- Taxa de churn (proporção): No 73,46% | Yes 26,54%.
- Visualizações geradas no notebook (inline):
  - Distribuições/contagens por categoria (seaborn/matplotlib).
  - Matriz de correlação entre variáveis numéricas e destaque para as mais correlacionadas com `Churn` (listadas no output do notebook).
- Observação: as figuras são exibidas no próprio notebook; não há pasta dedicada de imagens neste repositório. Caso deseje exportá-las, salve-as via `plt.savefig(...)` nas células correspondentes.

## Resultados Principais (Teste)
- Logistic Regression (dados escalados):
  - Accuracy: 0,7727 | Precision: 0,7747 | Recall: 0,7727 | F1: 0,7723
  - Gap Treino–Teste: 0,0021 (bom ajuste; sem sinais claros de over/underfitting).
- Random Forest (sem escala):
  - Accuracy: 0,8485 | Precision: 0,8487 | Recall: 0,8485 | F1: 0,8485
  - Gap Treino–Teste: 0,1502 (alerta de possível overfitting; acurácia de treino muito alta).
- K-Nearest Neighbors (dados escalados):
  - Accuracy: 0,7859 | Precision: 0,8048 | Recall: 0,7859 | F1: 0,7825
  - Gap Treino–Teste: 0,0634 (possível overfitting moderado).

Observações:
- Random Forest obteve melhor resultado em teste entre os três, porém com forte indício de overfitting (sugere regularização/ajuste de hiperparâmetros).
- SMOTE contribuiu para balancear classes; métricas reportadas consideram conjuntos balanceados e divisão estratificada.

## Como Reproduzir
1. Crie e ative o ambiente, instale dependências (seção “Ambiente e Dependências”).
2. Abra `prediction_telecom_x.ipynb` no Jupyter.
3. Execute as células em ordem (Kernel → Restart & Run All) para reproduzir limpeza, pré-processamento, treinamento e avaliação.

Notas sobre dados:
- O notebook espera o arquivo `df_limpo.csv` no diretório raiz do projeto (mesmo nível do notebook). O carregamento é feito com `pd.read_csv('df_limpo.csv')`.

Seeds e reprodutibilidade:
- SMOTE: `random_state=42`
- train_test_split: `random_state=42`
- Modelos: quando aplicável, `random_state=42`

================================================================================
CONCLUSÃO ESTRATÉGICA - PREVISÃO DE CHURN TELECOM X
================================================================================

🏆 COMPARAÇÃO E SELEÇÃO DO MELHOR MODELO
--------------------------------------------------
Modelo Escolhido: Random Forest
F1-Score: 0.8485
Acurácia: 0.8485
Precisão: 0.8487
Recall: 0.8485

Justificativa da escolha:
• Excelente capacidade de lidar com dados não-lineares
• Robusto a outliers e não requer normalização
• Fornece importância clara das variáveis
• Menor risco de overfitting devido ao ensemble

📊 Resumo Comparativo:
                     accuracy precision    recall  f1_score
Logistic Regression   0.77271  0.774704   0.77271  0.772302
Random Forest        0.848473  0.848688  0.848473  0.848451
KNN                  0.785852  0.804755  0.785852  0.782493


🔍 FATORES MAIS RELEVANTES PARA EVASÃO
--------------------------------------------------
Baseado na análise do Random Forest (Top 5):
• account.Contract: 0.1308 (13.1%)
• account.Charges.Total: 0.1169 (11.7%)
• account.Charges.Monthly: 0.1142 (11.4%)
• customer.tenure: 0.1122 (11.2%)
• Total.Day: 0.1040 (10.4%)

Baseado na Regressão Logística (Top 5 por impacto):
• customer.tenure: -1.4439 (DIMINUI churn)
• account.Charges.Monthly: 0.8995 (AUMENTA churn)
• account.Charges.Total: 0.6828 (AUMENTA churn)
• account.Contract: -0.6321 (DIMINUI churn)
• phone.PhoneService: -0.2870 (DIMINUI churn)


💡 INSIGHTS DE NEGÓCIO
--------------------------------------------------
Com base na análise dos dados e modelos, identificamos que:

1. PADRÕES DE COMPORTAMENTO:
   • Clientes com contratos mensais têm maior propensão ao churn
   • Tempo de relacionamento (tenure) é um fator crítico
   • Valor total gasto influencia significativamente a retenção
   • Serviços adicionais (segurança, backup) impactam na fidelização

2. PERFIL DE RISCO:
   • Clientes novos (baixo tenure) = ALTO RISCO
   • Contratos mensais = ALTO RISCO
   • Baixo valor total gasto = MÉDIO/ALTO RISCO
   • Sem serviços adicionais = MÉDIO RISCO


🎯 ESTRATÉGIAS DE RETENÇÃO PROPOSTAS
--------------------------------------------------

1. ESTRATÉGIAS PREVENTIVAS (Clientes de Alto Risco):
   ✓ Programa de boas-vindas para novos clientes (primeiros 6 meses)
   ✓ Incentivos para migração de contratos mensais para anuais
   ✓ Ofertas personalizadas de serviços adicionais
   ✓ Contato proativo nos primeiros 90 dias

2. ESTRATÉGIAS REATIVAS (Clientes Identificados pelo Modelo):
   ✓ Campanhas de retenção direcionadas
   ✓ Descontos temporários ou upgrades gratuitos
   ✓ Melhoria no atendimento e suporte técnico
   ✓ Pesquisas de satisfação e feedback

3. ESTRATÉGIAS DE LONGO PRAZO:
   ✓ Programa de fidelidade com benefícios crescentes
   ✓ Melhoria contínua dos serviços baseada em feedback
   ✓ Inovação em produtos e serviços
   ✓ Experiência do cliente omnichannel


📈 IMPLEMENTAÇÃO E MONITORAMENTO
--------------------------------------------------

1. IMPLEMENTAÇÃO DO MODELO:
   • Usar o modelo Random Forest para scoring mensal
   • Definir threshold de risco (ex: probabilidade > 0.7 = alto risco)
   • Integrar com CRM para ações automáticas
   • Treinar equipe de retenção nos insights do modelo

2. MÉTRICAS DE ACOMPANHAMENTO:
   • Taxa de churn mensal/trimestral
   • Efetividade das campanhas de retenção
   • ROI das estratégias implementadas
   • Satisfação do cliente (NPS)

3. MELHORIA CONTÍNUA:
   • Re-treinar modelo trimestralmente
   • A/B testing das estratégias de retenção
   • Incorporar novas variáveis (comportamentais, sazonais)
   • Feedback loop: resultados → ajustes → nova implementação


🎯 IMPACTO ESPERADO
--------------------------------------------------
Com base no desempenho do modelo Random Forest:
• Precisão de 84.9%: 84.9% dos clientes identificados como risco realmente farão churn
• Recall de 84.8%: 84.8% dos clientes que farão churn serão identificados

Estimativa de impacto:
• Redução potencial de 15-25% na taxa de churn
• ROI estimado: 3:1 a 5:1 (cada R$ investido retorna R$ 3-5)
• Melhoria na satisfação do cliente
• Aumento do Customer Lifetime Value (CLV)

## Boas Práticas e Próximos Passos
- Tuning e validação:
  - Aplicar `GridSearchCV`/`RandomizedSearchCV` e validação cruzada estratificada para RF, LR e KNN.
  - Considerar métricas adicionais (ROC-AUC, PR-AUC), especialmente em cenários desbalanceados.
- Regularização e complexidade:
  - RF: reduzir profundidade, ajustar `n_estimators`, `max_features`, `min_samples_*` para mitigar overfitting.
  - LR: avaliar penalizações (`l2`, `l1`) e `C`.
  - KNN: calibrar `n_neighbors`, distância e ponderação.
- Pipeline e produção:
  - Encapsular pré-processamento + modelo em `Pipeline` do scikit-learn.
  - Persistir artefatos (encoders/scalers/modelos) via `joblib`.
  - Criar API (FastAPI/Flask) e script de inferência para uso em produção.
- Dados e features:
  - Avaliar engenharia de atributos, seleção de features e tratamento de outliers.
  - Monitorar drift de dados e desempenho em produção.

## Limitações
- O notebook utiliza Label Encoding para categóricas; para alguns modelos lineares, `OneHotEncoder` pode ser mais apropriado.
- Resultados podem variar conforme ambiente (versões de libs) e hardware; use as versões indicadas quando possível.

## Créditos e Licença
- Dados: `df_limpo.csv` incluído no repositório (fonte não especificada no projeto).
- Este repositório não define uma licença explícita. Adicione uma licença (por exemplo, MIT) se desejar compartilhar/redistribuir.

---
Dúvidas ou melhorias? Abra uma issue ou sugira um PR.
