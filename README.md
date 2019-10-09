# Regressão Logística

## dataset 1: consumo de combustível
fonte do dataset: https://archive.ics.uci.edu/ml/datasets/Auto+MPG

### atributos
1. mpg: contínuo (milhas por galão)
2. cilindros: discreto multivalor
3. cilindradas: contínuo
4. cavalos: contínuo
5. peso: contínuo
6. aceleração: contínuo
7. ano: discreto multivalor (2 últimos dígitos)
8. origem: discreto multivalor (1 = USA, 2 = Europa, 3 = Japão)
9. nome do carro: categórico

### objetivo
estimar, dado os atributos covariantes 2-8, se um carro está acima da média em economia de combustível (mpg>=23) ou não

## dataset 2: dota 2
fonte do dataset: https://archive.ics.uci.edu/ml/datasets/Dota2+Games+Results

### atributos
1. quem ganhou o jogo (1 = próprio time, -1 = adversário)
2. cluster ID (servidor onde o jogo aconteceu, irrelevante para observação?)
3. modo de jogo (ex. "All Pick")
4. tipo de jogo (ex. "Ranked")

as colunas 5 em diante se referem a cada personagem selecionável no jogo, 113 no total. cada observação terá 5 dessas colunas com 1 e 5 com -1, indicando se faz parte do próprio time ou do time adversário.

### objetivo
estimar, dado os atributos de modo e tipo de jogo, e composição das equipes, se a própria equipe ganha ou perde