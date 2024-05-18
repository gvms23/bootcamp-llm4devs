# LLM4Devs - Aplicação prática Gabriel Vicente


## Comandos usados:
- Ambiente: Windows
- PostgreSQL na máquina + pgAdmin
- Docker:
```
docker run --name container-pgvector -e POSTGRES_PASSWORD=postgres -e PGUSER=postgres -d -p 5433:5432 pgvector/pgvector:pg16
```
Subi com a porta 5433 porque da 5432 o PostgreSQL na máquina já usava e precisava acessar o do Docker via pgAdmin.

-------------------------------

Olá! Que honra ter você conosco no Bootcamp LLM4Devs! Espero que o bootcamp seja muito útil na sua jornada na criação de aplicações baseadas em LLMs.

Antes de começarmos, vamos nos certificar que você com seu ambiente pronto para o bootcamp -- e assim focamos nosso tempo de sábado para as atividades *hands on*.

## Ferramentas

Para esse bootcamp, iremos utilizar Python como linguagem de programação e o Postgres como banco de dados. Certifique-se que você conseguiu instalar e configurar ambos. Versões recomendadas:

- Python: 3.10 ou superior
- PostgreSQL: 15 ou superior

Todas as demais dependencias serão instaladas via `pip`.

## Crie um ambiente virtual e instale as dependencias do Python

Para não criar uma confusão com as bibliotecas do seu sistema e as que serão utilizadas no bootcamp, é recomendado criar um ambiente virtual separado.

```
python3 -m venv env
source env/bin/activate
```

Em seguida, instale as bibliotecas: 

```
pip install -r requirements.txt
```

## Instale a extensão pgvector do Postgres

O PGVector é uma ferramenta muito importante para criação de aplicações baseadas em LLMs, uma vez que ele fornece novos tipos de armazenamento e busca de dados. Para instalá-lo, siga as instruções abaixo.

Para usuários Linux, o pgvector pode ser instalado via APT:

```
sudo apt install postgresql-16-pgvector
```

Para usuários macOS, pode-se utilizar o homebrew: 

```
brew install pgvector
```

Para demais métodos de instalação, consulte o [repositório oficial](https://github.com/pgvector/pgvector).

### Crie uma base de dados no postgres

Uma vez instalado, crie uma base de dados e uma tabela usando o tipo `VECTOR`. Primeiro, se conecte via linha de comando no seu banco de dados;


```bash
$ psql -h 127.0.0.1 -U postgres
```

Uma vez conectado, rode o script abaixo para criar a base e a tabela.


```sql 
CREATE DATABASE bootcamp_llm;

\c bootcamp_llm

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS embeddings;

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    chars INTEGER,
    embeddings VECTOR
);
```

## Configure acesso a OpenAI

Muito do nosso trabalho será baseado nas APIs da OpenAI. Para isso, você precisa criar uma chave para acessar a [plataforma da OpenAI](https://platform.openai.com/). 

Em seguida, adicione a variavel de ambiente `OPENAI_API_KEY` dentro do arquivo `.env`.

```
OPENAI_API_KEY = "<CHAVE>"
```

## Rode a interface web

Por fim, para garantir que a configuracao foi bem sucedida, rode o seguinte comando:

- `python3 $PATH/bootcamp-llm4devs/backend/core/main.py`

