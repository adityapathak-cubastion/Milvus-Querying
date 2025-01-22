# Importing libraries

from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

### Function Definitions

def connect_to_milvus():
    try:
        connections.connect("default", host = "localhost", port = "19530")
        print("\nMESSAGE : Connected to Milvus successfully!")
    except Exception as e:
        print(f"\nERROR : Failed to connect to Milvus!\nError: {e}")
        raise
def create_collection(name, fields, description):
    schema = CollectionSchema(fields, description)
    collection = Collection(name, schema, consistency_level = "Strong")
    print("MESSAGE : Collection created!")
    return collection
def insert_data(collection, entities):
    result = collection.insert(entities)
    collection.flush()
    print(f"MESSAGE : Inserted data into '{collection.name}'. Number of entities: {collection.num_entities}.")
    return result
def create_index(collection, field_name, index_type, metric_type, params):
    index = {"index_type": index_type,
             "metric_type": metric_type,
             "params": params
            }
    collection.create_index(field_name, index)
    print(f"MESSAGE : Index '{index_type}' created for field '{field_name}'.")
def search_and_query(collection, search_vectors, search_field, search_params):
    collection.load()
    
    result = collection.search(search_vectors, search_field, search_params,
                               limit = 3, output_fields = ["pk", "introduction", "age", "salary", "address"])
    print("\nRESULTS : Here are the vector seach results:")
    
    for hits in result:
        for i, hit in enumerate(hits):
            # print(f"Hit: {str((1 - round(hit.distance, 2))*100) + "%"} | Source Field: {hit.entity.get('source')}")
            print(f"{i + 1}) Distance: {round(hit.distance, 2)} | ID: {hit.entity.get('pk')} | Introduction: {hit.entity.get('introduction')}\n\
Age: {hit.entity.get('age')} | Salary: {hit.entity.get('salary')} | Address: {hit.entity.get("address")}")

### Main

connect_to_milvus()
fields = [
    FieldSchema(name = "pk", dtype = DataType.INT64, is_primary = True, auto_id = False, max_length = 10),
    FieldSchema(name = "introduction", dtype = DataType.VARCHAR, max_length = 400),
    FieldSchema(name = "age", dtype = DataType.INT64, max_length = 3),
    FieldSchema(name = "salary", dtype = DataType.INT64, max_length = 10),
    FieldSchema(name = "address", dtype = DataType.VARCHAR, max_length = 100),
    FieldSchema(name = "embeddings", dtype = DataType.FLOAT_VECTOR, dim = 384)
]
collection = create_collection("HelloMilvus", fields, "Demo Collection")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

introductions = ["Alice Johnson enjoys painting and hiking in her free time. She works as a software engineer at a tech startup in San Francisco.",
                 "Brian Smith loves playing the guitar and cycling. He is a marketing manager at a digital advertising agency in New York.",
                 "Catherine Lee is passionate about cooking and reading mystery novels. She works as a nurse at a hospital in Chicago.",
                 "David Kim enjoys playing basketball and video games. He is a financial analyst at an investment firm in Los Angeles.",
                 "Emily Davis loves gardening and practicing yoga. She works as a teacher at an elementary school in Boston.",
                 "Frank Miller is an avid photographer and enjoys traveling. He is a project manager at a construction company in Seattle.",
                 "Grace Wilson enjoys knitting and bird watching. She works as a librarian at a public library in Denver.",
                 "Henry Brown loves playing chess and running marathons. He is a lawyer at a law firm in Washington, D.C.",
                 "Isabella Martinez is passionate about dancing and baking. She works as a graphic designer at a creative agency in Miami.",
                 "Jack Thompson enjoys fishing and woodworking. He is an accountant at a financial services company in Dallas.",
                 "Karen Patel enjoys pottery and playing tennis. She works as a data scientist at a healthcare company in London.",
                 "Liam Nguyen loves hiking and playing the piano. He is a software developer at a gaming company in Toronto.",
                 "Mia Rodriguez is passionate about photography and volunteering. She works as a social worker at a non-profit organization in Sydney.",
                 "Noah Anderson enjoys woodworking and playing soccer. He is a mechanical engineer at an automotive company in Detroit.",
                 "Olivia Martinez loves painting and practicing meditation. She works as a marketing coordinator at a fashion brand in Paris.",
                 "Paul Harris is an avid reader and enjoys playing chess. He is a journalist at a news agency in Berlin.",
                 "Quinn Taylor enjoys cooking and rock climbing. He works as a chef at a restaurant in San Francisco",
                 "Rachel Green loves gardening and writing poetry. She is a content creator at a media company in New York.",
                 "Samuel Lee is passionate about cycling and playing the violin. He works as a research scientist at a university in Tokyo.",
                 "Tina Brown enjoys knitting and bird watching. She is a customer service representative at a retail company in Melbourne.",
                 "Uma Singh enjoys yoga and painting landscapes. She works as a human resources manager at a tech company in Bengaluru.",
                 "Victor Chen loves playing the violin and hiking. He is a civil engineer at an infrastructure firm in Singapore.",
                 "Wendy Parker is passionate about baking and reading historical fiction. She works as a pharmacist at a hospital in Toronto.",
                 "Xavier Johnson enjoys playing basketball and coding. He is a cybersecurity analyst at a financial institution in London.",
                 "Yara Ahmed loves gardening and practicing mindfulness. She works as a psychologist at a mental health clinic in Cairo.",
                 "Zachary Thompson is an avid runner and enjoys playing the guitar. He is a sales executive at a software company in New York.",
                 "Aisha Khan enjoys pottery and bird watching. She works as a graphic designer at an advertising agency in Dubai.",
                 "Benito Garcia loves playing soccer and cooking Italian cuisine. He is a chef at a popular restaurant in Madrid.",
                 "Clara Evans is passionate about photography and hiking. She works as a travel blogger and content creator based in Cape Town.",
                 "Dylan Brooks enjoys playing chess and writing short stories. He is an editor at a publishing house in Melbourne.",
                 "Eleanor White enjoys knitting and playing the piano. She works as a financial advisor at a bank in Dublin,",
                 "Felix Brown loves mountain biking and photography. He is a software engineer at a tech company in Austin.",
                 "Gabriella Rossi is passionate about cooking and learning new languages. She works as a translator at an international organization in Rome.",
                 "Hugo Martinez enjoys playing soccer and reading science fiction. He is a mechanical engineer at an aerospace company in Seattle.",
                 "Isla Thompson loves painting and practicing yoga. She works as a marketing specialist at a wellness brand in Los Angeles.",
                 "Jackie Lee is an avid runner and enjoys playing the guitar. She is a project manager at a construction firm in Vancouver.",
                 "Kevin Patel enjoys playing cricket and coding. He works as a data analyst at a financial services company in Mumbai.",
                 "Lila Green loves gardening and writing poetry. She is a teacher at a high school in Sydney.",
                 "Marcus Johnson is passionate about hiking and playing chess. He works as a lawyer at a law firm in Chicago.",
                 "Nina Kim enjoys baking and practicing meditation. She is a nurse at a hospital in Seoul.",
                 "Olivia Carter enjoys painting and playing the violin. She works as a marketing manager at a tech company in San Francisco.",
                 "Peter Wang loves hiking and playing chess. He is a software developer at a gaming company in Toronto.",
                 "Quincy Adams is passionate about cooking and reading science fiction. He works as a librarian at a public library in Boston.",
                 "Rachel Kim enjoys yoga and photography. She is a graphic designer at a creative agency in Los Angeles.",
                 "Samuel Johnson loves playing soccer and coding. He works as a data scientist at a healthcare startup in New York.",
                 "Tina Patel is an avid gardener and enjoys knitting. She is a teacher at an elementary school in London.",
                 "Umar Ahmed enjoys playing basketball and writing poetry. He works as a journalist at a news agency in Dubai.",
                 "Victoria Lee loves baking and practicing meditation. She is a nurse at a hospital in Sydney.",
                 "William Brown is passionate about woodworking and playing the guitar. He is a project manager at a construction company in Chicago.",
                 "Xena Martinez enjoys rock climbing and painting. She works as an environmental scientist at a research institute in Vancouver."
                ]
pks = [i for i in range(len(introductions))]
ages = [51, 43, 43, 53, 56, 48, 50, 57, 40, 40, 49, 25, 35, 56, 32, 55, 60, 27, 25, 55, 45,\
        41, 31, 34, 22, 60, 26, 41, 29, 34, 51, 43, 43, 53, 56, 48, 50, 57, 42, 56, 49, 37,\
        55, 47, 25, 29, 56, 51, 35, 57]
salaries = [61000, 115000, 82000, 80000, 51000, 75000, 50000, 58000, 60000, 49000, 73000, 89000,\
            104000, 110000, 104000, 106000, 108000, 109000, 105000, 113000, 101000, 103000, 104000,\
            108000, 107000, 109000, 104000, 103000, 104000, 108000, 61000, 115000, 82000, 80000, 51000,\
            75000, 50000, 58000, 115000, 86000, 53000, 98000, 60000, 91000, 65000, 114000, 73000, 64000,\
            61000, 105000]
addresses = ["123 Maple St, San Francisco, CA", "456 Oak Ave, New York, NY", "789 Pine Rd, Chicago, IL", "101 Elm St, Los Angeles, CA",\
             "202 Birch Ln, Boston, MA", "303 Cedar Dr, Seattle, WA", "404 Spruce St, Denver, CO", "505 Willow Ave, Washington, D.C.",\
             "606 Aspen Rd, Miami, FL", "707 Redwood St, Dallas, TX", "808 Palm Ave, London, UK", "909 Fir St, Toronto, ON", "1010 Cypress Ln, Sydney, NSW",\
             "1111 Maple St, Detroit, MI", "1212 Oak Ave, Paris, FR", "1313 Pine Rd, Berlin, DE", "1414 Elm St, San Francisco, CA", "1515 Birch Ln, New York, NY",\
             "1616 Cedar Dr, Tokyo, JP", "1717 Spruce St, Melbourne, VIC", "1818 Willow Ave, Bengaluru, IN", "1919 Aspen Rd, Singapore, SG", "2020 Redwood St, Toronto, ON",\
             "2121 Palm Ave, London, UK", "2222 Fir St, Cairo, EG", "2323 Cypress Ln, New York, NY", "2424 Maple St, Dubai, AE", "2525 Oak Ave, Madrid, ES",\
             "2626 Pine Rd, Cape Town, ZA", "2727 Elm St, Melbourne, VIC", "2828 Birch Ln, Dublin, IE", "2929 Cedar Dr, Austin, TX", "3030 Spruce St, Rome, IT",\
             "3131 Willow Ave, Seattle, WA", "3232 Aspen Rd, Los Angeles, CA", "3333 Redwood St, Vancouver, BC", "3434 Palm Ave, Mumbai, IN", "3535 Fir St, Sydney, NSW",\
             "3636 Cypress Ln, Chicago, IL", "3737 Maple St, Seoul, KR", "3838 Oak Ave, San Francisco, CA", "3939 Pine Rd, Toronto, ON", "4040 Elm St, Boston, MA",\
             "4141 Birch Ln, Los Angeles, CA", "4242 Cedar Dr, New York, NY", "4343 Spruce St, London, UK", "4444 Willow Ave, Dubai, AE", "4545 Aspen Rd, Sydney, NSW",\
             "4646 Redwood St, Chicago, IL", "4747 Palm Ave, Vancouver, BC"]

embeddings = model.encode(introductions)
print("MESSAGE : Embeddings generated!")

entities = [
            [int(i) for i in pks],
            [str(intro) for intro in introductions],
            [int(age) for age in ages],
            [int(salary) for salary in salaries],
            [str(address) for address in addresses],
            embeddings
           ]

result = insert_data(collection, entities)

create_index(collection, "embeddings", "IVF_FLAT", "COSINE", {"nlist" : 128})

while True:
    query = str(input("\nUSER-INPUT : Enter your query! To exit, type 'exit': "))
    if query == "exit" or query == "Exit" or query == "EXIT":
        break
    query_vector = model.encode(query)
    search_and_query(collection, [query_vector], "embeddings",
                     {"metric_type" : "COSINE", "params" : {"nprobe" : 16}})

utility.drop_collection("HelloMilvus")
print("\nMESSAGE : Dropped collection 'HelloMilvus'.")