from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.llms import Ollama
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
url="url"
username="neo4j"
password="password"
graph = Neo4jGraph(url=url, username="neo4j", password=password)
embedding_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
def properties_to_embedding(properties):
    dict=properties.copy()
    dict.pop("label")
    combined_text = ' '.join([str(value) for value in dict.values()])
    embedding = embedding_func.embed_query(combined_text)
    return embedding
def node_exist(graph, label, properties):
    name=properties['name']    
    query = f"MATCH (n:{label}) WHERE n.name='{name}' RETURN n"
    response = graph.query(query)
    
    if len(response) == 0:
        return False
    else:
        return True
def create_node(graph, label, properties):
    properties["Embedding"]=properties_to_embedding(properties)
    if not(node_exist(graph,label,properties)):
        query = f"CREATE (n:{label} $properties)"
        graph.query(query,params={'properties':properties})
        return 1
    else:
        return 0
def node_relation(graph, label1, properties1,relation,label2,properties2):
    name1=properties1['name']
    name2=properties2['name']
    query=f"""MATCH (n:{label1} ),(m:{label2})
              WHERE n.name='{name1}' and m.name='{name2}'
             CREATE (n)-[:{relation}]->(m)"""
    
    graph.query(query)
    return True
def node_create_relate(graph, label1, properties1,relation,label2,properties2):
    create_node(graph,label1,properties1)
    create_node(graph,label2,properties2)
    node_relation(graph, label1, properties1,relation,label2,properties2)
    return True
def node_retrever(question):
    question_embedding= embedding_func.embed_query(question)
    node=graph.query("""CALL db.index.vector.queryNodes(
        'node_index', 
        3, 
        $question_embedding
        ) YIELD node AS node, score
    RETURN node, score""",params=({'question_embedding':question_embedding}))
    for item in node:
        if 'Embedding' in item['node']:
            del item['node']['Embedding']
    return node

def neighbour_relation(question):
    node=node_retrever(question)
    result=[]
    for item in node:
        properties=item["node"]
        name=properties['name']
        query="""
                MATCH (n:node)-[r]->(m)
                WHERE n.name=$name
                RETURN n.name+'-'+type(r)+'->'+m.name AS Output
                UNION
                MATCH (n:node)<-[r]-(m)
                WHERE n.name=$name
                RETURN m.name+'-'+type(r)+'->'+n.name AS Output
                """
        a=graph.query(query,params={'name':name})
        result.append(a)
    return result
def create_context(neo4j_data):
    context = ""
    for data in neo4j_data:
        for entry in data:
            text= f"{entry['Output'].replace('-',' ')}\n"
            text=text.replace('>','')
            text=text.replace('_',' ')
            context+=text
    return context

template = """Answer the question based only on the following information:
{context}

Question: {question}
"""

prompt_template=PromptTemplate(input_variables=['context','question'],template=template)
model=Ollama(model='llama3')
RAG_chain=LLMChain(llm=model,prompt=prompt_template,output_key="output",verbose=True)
def RAG(question):
    data=neighbour_relation(question)
    context=create_context(data)
    result=RAG_chain({'context':context,'question':question})
    output=result['output']
    return output

    
