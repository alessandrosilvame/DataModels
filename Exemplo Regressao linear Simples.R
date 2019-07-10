

#----Exemplo Geral Regress?o Linear Simples----

#Comando para tratar sempre do mesmo DataSet.
attach(RLS_BEZERRO)

# Plot para analisar a dispersao com ajuste de cor e fontes. 
plot(`Medida(CM)`,`Peso (KG)`, pch = 19, col = "Dark Blue", main = "Medida x Peso",
     cex.axis = 1.5, cex.main = 1.5, cex.lab = 1.5, cex = 1.5)
cor.test(`Medida(CM)`,`Peso (KG)`)

# Ajuste do modelo de regressao linear.
ajuste <- lm(`Peso (KG)` ~ `Medida(CM)`)

# Analise do Ajuste.
summary(ajuste)

# Procura por Outliers via BoxPlot com fontes e cores personalizaadas.
boxplot(`Medida(CM)`, col = "Light Blue", main = "Box Plot Medida (CM)",
        cex.axis = 1.5, cex.main = 1.5, cex.lab = 1.5)
boxplot(`Peso (KG)`, col = "Light Green", main = "Box Plot Peso (KG)",
        cex.axis = 1.5, cex.main = 1.5, cex.lab = 1.5)

# Padronização Z para Medida e PEso.
RLS_BEZERRO$Z_Medida <- scale(`Medida(CM)`)
RLS_BEZERRO$Z_Peso <- scale(`Peso (KG)`)

# Analise de Range de Peso e Medida
# Esta deve estar sempre entre -3 e 3 caso o contrario estacom problemas.
range(RLS_BEZERRO$Z_Medida)
range(RLS_BEZERRO$Z_Peso)

# Cria uma matriz de graficos
par(mfrow=c(2,2))
# Plot os graficos padroes dos ajustes
plot(ajuste)
# Os Residuals vs Fitted devem estar bem distribuidos e espacados para 
# estarem corretos, caso o contrario a amostra é tendenciosa a erro no
# modelo. 

# A Scale Location segue a mesma visao dos Residuos
# A normal Q-Q deve ter os pontos o mais proximo possiveis da reta, e assim
# estao adequados, podem haver pequenos desvios se forem grandes o modelo
# pode nao responder corretamente. 

# Residuals vs Leverage deve ser sempre o mais proximo de 0, assim mantendo
# o modelo adequado, se o desvio de 0 for muito distante existem problemas.

# Na variavel predicao, apresento os valores a serem previstos, ou seja, o 
# peso neste caso de 20Kg e 28Kg.
predicao <- data.frame(Peso_previsto = c(20, 28))
predicao

# Na variavel Coef_Ajuste, se armazena os coeficients encontrados no ajuste.
coef_ajuste <- coefficients(ajuste)

# Calculo usando os valores de predicao com os valores de peso previsto, 
# utilizando respectivamente o BETA0 e BETA1, sendo eles coef_ajuste[1] e 
# coef_ajuste[2] * os valores de predicao para prever o peso do bezzero. 
predicao$Peso_Previsto <- coef_ajuste[1]+coef_ajuste[2]*predicao

# Concatena via colnames no dataframe predicao as colunas Medida e Peso
# Previsto. 
colnames(predicao) <- c("Medida", "Peso_previsto")

# Apresenta o resultado. 
predicao


