import os
import json
from neo4j import GraphDatabase
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

from dotenv import load_dotenv

load_dotenv()

class KnowledgeGraph:
    """This class handles the creation and management of a knowledge graph using Neo4j."""
    def __init__(self) -> None:

        # we load info about the database from the .env file @Luca
        self.URI = os.getenv('NEO4J_URI')
        self.AUTH = (os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))

        self.driver = GraphDatabase.driver(self.URI, auth=self.AUTH)
        try:
            self.driver.verify_connectivity()
            print("[Knowledge Graph] Connected to the online DB ✅")
        except Exception as e:
            print(f"\033[91m[Knowledge Graph] An error occurred while connecting to the online DB: {e}\033[0m")

        self.graph = None
        
        # Dictionary to map English to Italian poses
        self.yoga_pose_aliases = {
            "Warrior": "Guerriero",
            "Airplane": "Aeroplano",
            "Crab": "Granchio",
            "Fighter": "Lottatore",
            "Tree": "Albero",
            "Flamingo": "Fenicottero",
            "Cactus": "Cactus",
            "Wheel": "Ruota",
            "Triangle": "Triangolo"
        }

        self.system_prompt = (
            "# Knowledge Graph Instructions for GPT-4\n"
            "## 1. Overview\n"
            "You are a top-tier algorithm designed for extracting information in structured "
            "formats to build a knowledge graph.\n"
            "Try to capture as much information from the text as possible without "
            "sacrificing accuracy. Do not add any information that is not explicitly "
            "mentioned in the text.\n"
            "- **Nodes** represent entities and concepts.\n"
            "- **Relationships** represent connections between entities or concepts.\n"
            "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
            "accessible for a vast audience.\n"
            "## 2. Labeling Nodes and Adding Properties\n"
            "- **Nodes**: Capture entities and concepts. Use simple and basic node types (e.g., 'Person', 'Activity').\n"
            "- **Node Properties**: Extract relevant properties for each node (e.g., name, emotional state, hobbies, level, etc.). Properties must be in a key-value format and use camelCase for keys (e.g., 'name', 'trainingLevel').\n"
            "- **Relationships**: Represent connections between entities or concepts. Use general, consistent, and timeless relationship types (e.g., 'GUIDED_BY', 'PRACTICES').\n"
            "Ensure consistency and generality in relationship types when constructing knowledge graphs."
            "- **Relationship Properties**: If possible, extract properties of the relationship (e.g., start date, type of feedback).\n"
            "- **Consistency**: Ensure you use available types for node labels.\n"
            "Ensure you use basic or elementary types for node labels.\n"
            "- For example, when you identify an entity representing a person, "
            "always label it as **'person'**. Avoid using more specific terms "
            "like 'mathematician' or 'scientist'."
            "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
            "names or human-readable identifiers found in the text.\n"
            "- **Quotation Marks**: Never use escaped single or double quotes within property values."
            "## 3. Handling Numerical Data and Dates\n"
            "- Attach numerical data and dates as attributes or properties of the respective nodes or relationships.\n"
            "- No separate nodes for dates or numbers."
            "## 4. Coreference Resolution\n"
            "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
            "ensure consistency.\n"
            'If an entity, such as "John Doe", is mentioned multiple times in the text '
            'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
            "always use the most complete identifier for that entity throughout the "
            'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
            "Remember, the knowledge graph should be coherent and easily understandable, "
            "so maintaining consistency in entity references is crucial.\n"
            "## 5. Output Format\n"
            "Your response should be structured as a JSON object with the following fields:\n"
            "1. **nodes**: List of extracted nodes with their types and properties.\n"
            "2. **relationships**: List of relationships with source and target nodes, relationship types, and any properties related to the relationship."
            "## 6. Strict Compliance\n"
            "Adhere to these rules strictly. Non-compliance will result in termination."
        )

        self.examples = [
            {
                "text": (
                    "Dev Patel is interested in Classic Cars."
                ),
                "node": "Dev Patel",
                "node_type": "Person",
                "relation": "IS_INTEREST_IN",
                "target_node": "Classic cars",
                "target_node_type": "Hobby",
            },
            {
                "text": (
                    "Giulio listens to classical music."
                ),
                "node": "Giulio",
                "node_type": "Person",
                "relation": "HAS_HOBBY",
                "target_node": "Classical music",
                "target_node_type": "Classical music",
            },
            {
                "text": (
                    "Particular Difficulties Encountered by the User: Difficulty with alignment in the albero pose."
                ),
                "node": "User",
                "node_type": "Person",
                "relation": "COMPLETED_POSE",
                "target_node": "Albero",
                "target_node_type": "Yoga_pose",
                "relationship_properties": [{"key": "feedback", "value": "Difficulty with alignment"}]
            },
            {
                "text": (
                    "Marco is an advanced climber who practices climbing every weekend"
                ),
                "node": "Marco",
                "node_type": "Person",
                "relation": "PRACTICES",
                "target_node": "Climbing",
                "target_node_type": "Sport",
                "node_properties": [{"key": "level", "value": "advanced"}],
                "relationship_properties": [{"key": "frequency", "value": "every weekend"}]
            },
            {
                "text": (
                    "Marco enjoys hiking in the summer."
                ),
                "node": "Marco",
                "node_type": "Person",
                "relation": "PRACTICES",
                "target_node": "Hiking",
                "target_node_type": "Sport",
                "relationship_properties": [{"key": "frequency", "value": "in the summer"}]
            },
            {
                "text": (
                    "Lorenzo didn't manage to complete the 'Warrior' pose and find it difficult."
                ),
                "node": "Lorenzo",
                "node_type": "Person",
                "relation": "COMPLETED_POSE",
                "target_node": "Warrior",
                "target_node_type": "Yoga_pose",
                "relationship_properties": [{"key": "feedback", "value": "difficult"}, {"key": "success", "value": "false"}]
            },
            {
                "text": (
                    "David loves playing chess during his free time, but he dislikes video games."
                ),
                "node": "David",
                "node_type": "Person",
                "relation": "PRACTICES",
                "target_node": "Chess",
                "target_node_type": "Hobby",
                "relationship_properties": [{"key": "frequency", "value": "free time"}]
            },
            {
                "text": (
                    "David loves playing chess during his free time, but he dislikes video games."
                ),
                "node": "David",
                "node_type": "Person",
                "relation": "DISLIKES",
                "target_node": "Video games",
                "target_node_type": "Hobby",
                "relationship_properties": []
            },
            {
                "text": (
                    "Sarah is a huge fan of the Star Wars movies."
                ),
                "node": "Sarah",
                "node_type": "Person",
                "relation": "IS_INTEREST_IN",
                "target_node": "Star Wars",
                "target_node_type": "Movie",
            },
            {
                "text": (
                    "Lorenzo Morrocotti expressed his feedback, suggesting that iCub could be stricter "
                    "during their yoga sessions."
                ),
                "node": "Lorenzo Morrocotti",
                "node_type": "Person",
                "relation": "HAS_FEEDBACK",
                "target_node": "iCub",
                "target_node_type": "Robot",
                "relationship_properties": [{"key": "feedback", "value": "be stricter"}]
            },
            {
                "text": (
                    "Giulia is feeling stressed because of her studies at university."
                ),
                "node": "Giulia",
                "node_type": "Person",
                "node_properties": [{"key": "EmotionalState", "value": "stressed"}, {"key": "cause_of_EmotionalState", "value": "studies at university"}]
            },
            {
                "text": (
                    "Lorenzo Morrocotti requested more challenging yoga poses, as he found the current session too easy."
                ),
                "node": "Lorenzo Morrocotti",
                "node_type": "Person",
                "relation": "HAS_FEEDBACK",
                "target_node": "iCub",
                "target_node_type": "Robot",
                "node_properties": [{"key": "feedback", "value": "current session too easy"}]
            },
            {
                "text": (
                    "Lorenzo Morrocotti mentioned having an intermediate training level."
                ),
                "node": "Lorenzo Morrocotti",
                "node_type": "Person",
                "relation": "TRAINING_LEVEL",
                "target_node": "Intermediate",
                "target_node_type": "Training level",
            },
            {
                "text": (
                    "Mario is frustrated by the fact he's not loosing weight - User Name: Mario rossi \n - User Language: italian \n - Training Level: intermediate"
                ),
                "node": "Mario Rossi",
                "node_type": "Person",
                "node_properties": [{"key": "EmotionalState", "value": "frustrated"},{"key": "cause_of_EmotionalState", "value": "not loosing weight"},{"key": "User Language", "value": "italian"}, {"key": "Training Level", "value": "intermediate"}]
            },
            {
                "text": (
                    "- Poses Practiced: ['aeroplano', 'granchio', 'guerriero'] \n - Poses Done Correctly: ['aeroplano', 'granchio']"
                ),
                "node": "Lorenzo Morrocotti",
                "node_type": "Person",
                "relation": "COMPLETED_POSE",
                "target_node": "Aeroplano, Granchio",
                "target_node_type": "Yoga_pose",
                "relationship_properties": [{"key": "success", "value": "true"}]
            },
            {
                "text": (
                    "- Poses Practiced: ['aeroplano', 'granchio', 'guerriero'] \n - Poses Done Correctly: ['aeroplano', 'granchio']"
                ),
                "node": "Lorenzo Morrocotti",
                "node_type": "Person",
                "relation": "COMPLETED_POSE",
                "target_node": "Guerriero",
                "target_node_type": "Yoga_pose",
                "relationship_properties": [{"key": "success", "value": "false"}]
            },
            {
                "text": (
                    "Lorenzo Morrocotti found the 'Fighter' pose easy and relaxing."
                ),
                "node": "Lorenzo Morrocotti",
                "node_type": "Person",
                "relation": "COMPLETED_POSE",
                "target_node": "Fighter",
                "target_node_type": "Yoga_pose",
                "relationship_properties": [{"key": "feedback", "value": "easy but relaxing"}]
            }
        ]

        self.examples_prompt = "Below are some examples to build the knowledge graph: {examples}"

        # Initialize the LLM and graph transformer
        self.llm = AzureChatOpenAI(
            openai_api_version="2024-10-21",
            deployment_name="contact-Yogaexperiment_gpt4omini",
            temperature=0,
        )

        self.graph_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are reading the summary of a yoga training session with the iCub robot. When creating the knowledge graph, pay particular attention on personal details like the job, personal interests, hobbies, emotional state, and feedback. \
                        If you create a 'Person' Node it should only contain the name of the person, its emotional state, the language and the corresponding training level. \
                        Use the relationship COMPLETED_POSE to connect people with the practiced poses, use the property 'success=true' to denote that the pose was completed successfully. \
                        Use the property 'success=false' to denote that the pose was not completed successfully. \
                        Read the poses done correctly and the poses practiced from the 'The user's most important information' section of the summary. \
                        Here's some examples to help you build the knowledge graph: " + self.examples_prompt +
                        "\n Here is the summary you have to use to extract the graph from: {input}"
                    ),
                ),
            ]
        ).partial(examples=self.examples)
        
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["Person", "Activity", "Yoga_pose", "Training level", "Interests", "Emotion", "Job", "Animal", "Object", "Robot", "Hobby", "Sport", "Organization", "Event"],
            allowed_relationships=["COMPLETED_POSE", "WORK_AS", "WORK_AT" "TRAINING_LEVEL", "PRACTICES", "IS_INTEREST_IN", "DISCUSSES", "LIKES", "DISLIKES" "OWNS", "IS_FRIEND_OF"],
            node_properties=True,
            relationship_properties=True,
            strict_mode=False,
            prompt=self.graph_prompt
        )

    def connect_to_database(self):
        """Establishing Neo4j connection."""
        self.driver.verify_connectivity()

    def disconnect_from_database(self):
        """Ensure the Neo4j driver is closed properly."""
        if 'driver' in locals():
            self.driver.close()
            print("Neo4j driver closed.")

    def get_content(self, file_path):
        """Reads content from the specified file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as input_file:
                content = input_file.read()
            return content
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"An error occurred while reading the summary: {e}")

    def merge_node(self, driver, label, id, properties):
        label = f"`{label}`"  # Enclose label in backticks

        with driver.session() as session:
            if label == "`Yoga pose`":
                # Handle aliases specifically for 'Yoga Pose' nodes
                aliases = [id]
                if id in self.yoga_pose_aliases:
                    aliases.append(self.yoga_pose_aliases[id])  # Add the Italian alias
                elif id in self.yoga_pose_aliases.values():
                    # If the id is an Italian alias, find the English equivalent
                    corresponding_english_id = [k for k, v in self.yoga_pose_aliases.items() if v == id][0]
                    aliases.append(corresponding_english_id)

                # Check if a node with this ID or alias already exists
                query = f"""
                MATCH (n:{label})
                WHERE n.id = $id OR ANY(alias IN $aliases WHERE alias IN n.aliases) OR $id IN n.aliases
                RETURN n
                """
                result = session.run(query, id=id, aliases=aliases)

                if result.peek():  # Node exists
                    existing_node = result.single()["n"]
                    existing_id = existing_node["id"]

                    # Handle transformation from English to Italian ID if needed
                    if id in self.yoga_pose_aliases.values():  
                        if existing_id not in self.yoga_pose_aliases.values():  
                            print(f"Changing current ID: {existing_id} with Italian ID: {id}")
                            session.run(f"""
                            MATCH (n:{label} {{id: $existing_id}})
                            SET n.id = $id, n.aliases = coalesce(n.aliases, []) + $existing_id
                            """, id=id, existing_id=existing_id)

                    # Update properties and aliases
                    session.run(f"""
                    MATCH (n:{label} {{id: $existing_id}})
                    SET n += $properties, n.aliases = coalesce(n.aliases, []) + $aliases
                    """, existing_id=existing_id, properties=properties, aliases=aliases)
                
                else:
                    # Create the node if it doesn't exist, with properties and aliases
                    session.run(f"""
                    MERGE (n:{label} {{id: $id}})
                    SET n += $properties, n.aliases = coalesce(n.aliases, []) + $aliases
                    """, id=id, properties=properties, aliases=aliases)
            
            else:
                # For other labels, do not include the aliases field
                query = f"""
                MATCH (n:{label} {{id: $id}})
                RETURN n
                """
                result = session.run(query, id=id)

                if result.peek():  # Node exists
                    existing_node = result.single()["n"]

                    # Update only the properties (no aliases for non-Yoga Pose nodes)
                    session.run(f"""
                    MATCH (n:{label} {{id: $id}})
                    SET n += $properties
                    """, id=id, properties=properties)
                
                else:
                    # Create the node with only properties (no aliases)
                    session.run(f"""
                    MERGE (n:{label} {{id: $id}})
                    SET n += $properties
                    """, id=id, properties=properties)

    def merge_relationship(self, driver, label_source, id_source, label_target, id_target, rel_type, properties):
        label_source = f"`{label_source}`"  # Enclose label in backticks
        label_target = f"`{label_target}`"  # Enclose label in backticks
        
        with driver.session() as session:
            query = f"""
            MATCH (a:{label_source} {{id: $id_source}})
            MATCH (b:{label_target} {{id: $id_target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $properties
            """
            session.run(query, id_source=id_source, id_target=id_target, properties=properties)

    def is_database_empty(self, driver):
        """ Check if the database is empty. """
        with driver.session() as session:
            # Run a query to count the number of nodes
            result = session.run("MATCH (n) RETURN count(n) AS node_count")
            node_count = result.single()["node_count"]
            
            # Check if the database is empty
            if node_count == 0:
                print("The database is empty.")
                return True
            else:
                print(f"The database contains {node_count} nodes.")
                return False

    def build_knowledge_graph(self, summary_path, raw_chat_path):
        """Builds a knowledge graph from the provided data:
            Args:
                summary_path (str): The path to the .txt summary file.
                raw_chat_path (str): The path to the .txt raw chat file.
        """

        # BUILD KNOWLEDGE GRAPH FROM RAW_CHAT.TXT
        # TODO: commented out, this should be considered in the future for ablations studies
        #self.build_graph_from_file(raw_chat_path)
        
        # BUILD KNOWLEDGE GRAPH FROM AND SUMMARY.TXT
        self.build_graph_from_file(summary_path)

    def build_graph_from_file(self, file_path):
        """Builds a knowledge graph from the provided file (e.g raw_chat.txt, summary.txt).
        Args:
            file_path (str): The path to the file containing the text data. 
        """
        print(f"[Knowledge Graph] Building knowledge graph from: {file_path} \n")
        try:
            self.connect_to_database()

            text = self.get_content(file_path)

            documents = [Document(page_content=text)]

            # Process documents and convert to graph documents
            graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
            print("---------------------------------------------------------------")
            print(f"Nodes: {graph_documents[0].nodes}")
            print("---------------------------------------------------------------")
            print(f"Relationships: {graph_documents[0].relationships}")
            print("---------------------------------------------------------------")

            for node in graph_documents[0].nodes:
                self.merge_node(self.driver, node.type, node.id, node.properties)

            for relationship in graph_documents[0].relationships:
                self.merge_relationship(
                    self.driver,
                    relationship.source.type, relationship.source.id,
                    relationship.target.type, relationship.target.id,
                    relationship.type, relationship.properties
                )
        finally:
            self.disconnect_from_database()

    def gather_person_info(self, person_name: str, max_token_limit: int = 2048) -> str:
        """
        Gathers all relevant information about a person from the knowledge graph.
        This includes past interactions, preferences, emotions, etc.
        """
        result = ""
        query = """
        MATCH (p:Person)
        WHERE toLower(p.id) = toLower($person_name)
        MATCH (p)-[r]->(neighbor)
        RETURN 
            p.id AS PersonName, 
            properties(p) AS PersonProperties, 
            type(r) AS RelationshipType, 
            properties(r) AS RelationshipProperties, 
            labels(neighbor) AS NeighborLabels,
            neighbor.id AS NeighborId, 
            properties(neighbor) AS NeighborProperties
        """
        
        response = self.graph.query(query, {"person_name": person_name})
        
        for record in response:
            person_name = record["PersonName"]
            person_props = ", ".join([f"{k}: {v}" for k, v in record["PersonProperties"].items()])
            relationship_type = record["RelationshipType"]
            relationship_props = ", ".join([f"{k}: {v}" for k, v in record["RelationshipProperties"].items()])
            neighbor_labels = ":".join(record["NeighborLabels"])
            neighbor_id = record["NeighborId"]
            neighbor_props = ", ".join([f"{k}: {v}" for k, v in record["NeighborProperties"].items()])
            
            # Check if adding this record exceeds the token limit
            potential_result = (
                f"Person ({person_name}) [{person_props}] -[{relationship_type} ({relationship_props})]-> "
                f"{neighbor_labels} ({neighbor_id}) [{neighbor_props}]\n"
            )

            token_count = len(result.split()) + len(potential_result.split())

            if token_count > max_token_limit:
                break

            result += potential_result

        return result

    def retrieve_all_the_knowledge(self):
        try:
            self.connect_to_database()

            knowledge = self.get_whole_graph()

            print("all the robot knowledge: ", knowledge)
            return knowledge
        
        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            self.close_db_connection()

    def retrieve_knowledge(self, user_name):
        try:
            self.connect_to_database()
            context = self.gather_person_info(user_name)

            print("memories: ", context)
            return context
        
        except Exception as e:
            print(f"An error occurred: {e}")

        #finally:
            #self.close_db_connection() not implemented

    def compute_performance_score(self, input_dict):
        success = None
        performance_score = None

        if not input_dict:
            return success, performance_score # Return None if the dictionary is empty
        
        # if the Time elapsed before success is None, this means that the user does not complete successfully the yoga pose
        elif input_dict['Time elapsed before success (in seconds)'] == 'None':
            success = False
            return success, performance_score
        else:
            success = True
            time_elapsed = float(input_dict['Time elapsed before success (in seconds)'])
            max_time = float(input_dict['Max exercise time (in seconds)'])

            if time_elapsed is not None and max_time is not None and max_time > 0:
                performance_score = (1 - (time_elapsed / max_time)) * 100
                return success, performance_score
            else:
                return success, performance_score

    def get_performance_score(self, metrics):

        average_performance_score = {}
        user_performance_score = {}

        for key, value in metrics.items():
            print()
            success, performance_score = self.compute_performance_score(metrics[key]['time_performance'])

            pose_name = key.split('_')[0]

            if success is not None and success is not False:
                user_performance_score[pose_name] = performance_score
            else:
                user_performance_score[pose_name] = "yoga pose not successfully completed"

            
            with self.driver.session() as session:
                
                # Define the Cypher query with dynamic yoga pose and performance score property
                query = f"""
                MATCH (y:`Yoga pose`)
                WHERE toLower(y.id) = toLower($pose_name)
                RETURN y.id AS yoga_pose,  y.`{pose_name} performance score average` AS performance_score
                """
                
                # Execute the query
                result = session.run(query, pose_name=pose_name)

                # Retrieve and return the performance score
                avg_performance_score = None
                for record in result:
                    avg_performance_score = record['performance_score']
                    print(f"Yoga Pose: {record['yoga_pose']}")
                    print(f"Performance Score: {avg_performance_score}")
                
                average_performance_score[pose_name] = avg_performance_score

        return user_performance_score, average_performance_score
    
    def get_whole_graph(self):
        """ 
        Get whole graph
        """

        enhanced_graph = Neo4jGraph(
                                    url=self.URI,
                                    username="neo4j",
                                    password=self.AUTH[1],
                                    enhanced_schema=True,
                                )
        
        # MATCH (n)-[r]->(m) RETURN n, r, m 
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN collect({
            nodeId: n.id,
            nodeLabels: labels(n),
            nodeProperties: n{.*},
            relationshipType: type(r),
            relationshipProperties: r{.*},
            neighborId: m.id,
            neighborLabels: labels(m),
            neighborProperties: m{.*}
        }) AS graphData
        """
        
        extracted_graph = enhanced_graph.query(query)

        return extracted_graph
    
    def get_personal_graph(self, username):
        """
        Gathers all nodes and relationships from the knowledge graph,
        including their properties, and returns them as a JSON-like structure.
        """
        query = """
        MATCH (n:Person)
        WHERE toLower(n.id) = toLower($person_name)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN collect({
            nodeId: n.id,
            nodeLabels: labels(n),
            nodeProperties: n{.*},
            relationshipType: type(r),
            relationshipProperties: r{.*},
            neighborId: m.id,
            neighborLabels: labels(m),
            neighborProperties: m{.*}
        }) AS graphData
        """

        enhanced_graph = Neo4jGraph(
                            url=self.URI,
                            username="neo4j",
                            password=self.AUTH[1],
                            enhanced_schema=True,
                        )
        
        retrieved_graph = enhanced_graph.query(query, {"person_name": username})

        return  retrieved_graph

    def ask_graph(self, graph, question):
        graph_chain = ChatPromptTemplate.from_template("You are a Yoga teacher. Answer the user's question by carefully assessing the provided GRAPH. \
                                                       Pay attention to the nodes and their relationships. \
                                                       Be as accurate and concise as possible. \
                                                       Generate an answer that mirrors the structure and phrasing of the question (without repeating it). Do not introduce additional words, synonyms, or unnecessary context. \
                                                       \n GRAPH: {graph} \n question: {question}") | self.llm
        answer = graph_chain.invoke({"graph": graph, "question": question}).content

        return answer
    
    def agentic_ask_graph(self, question):
        if not self.graph:
            self.graph = Neo4jGraph(
                                url=self.URI,
                                username="neo4j",
                                password=self.AUTH[1],
                                enhanced_schema=True,
                            )
        
        CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
                                        Instructions:
                                        Use only the provided relationship types and properties in the schema.
                                        Do not use any other relationship types or properties that are not provided.
                                        Schema:
                                        {schema}
                                        Do not include any text except the generated Cypher statement.
                                        NEVER include in the Cypher statement any node label that is not present in the Schema.


                                        
                                        You are a Yoga teacher. Answer the user's question by carefully assessing the provided graph.
                                        In the graph you have the nodes of the people who interacted with you (the robot iCub), the yoga poses that they did, and the relationships between them.
                                        Each person node has a name, an emotional state, a language, and a training level (beginner, intermediate, advanced).
                                        Multiple people can share the same properties so you should consider the whole population of users when replying to a question. e.g. Who is the person who is feeling stressed? -> return all the people who are feeling stressed.
                                        The relationships between the person nodes and the yoga pose nodes are labeled as COMPLETED_POSE.
                                        The COMPLETED_POSE relationship has a property called success that can be either true or false.
                                        The user's most important information is the poses practiced and the poses done correctly.
                                        The poses practiced are the yoga poses that the user has tried to do.
                                        The poses done correctly are the yoga poses that the user has successfully completed.

                                        Adapt the ids of the query to the schema provided: e.g. iCub -> Icub, beginner -> Beginner, hiking -> Hiking, etc.
                                        Try instead to adapt the query to the user's question.  e.g. if the user ask if someone is interested in books and in the schema there is the node "Reading" you should look for the node "Reading" in the query. 
                                        Other examples "Read"->Reading, "Social_Events"->Parties, "Other cultures"-> "Traveling","Travelling", "Travel", "Reading about other cultures" etc. Be as generative as possible, for example:
                                        Who appeared in the Film 'Top Gun'?
                                        ( Schema:
                                            Node properties:
                                            - **Movie**
                                            - `runtime`: INTEGER Min: 120, Max: 120
                                            - `name`: STRING Available options: ['Top Gun']
                                            - **Actor**
                                            - `name`: STRING Available options: ['Tom Cruise', 'Val Kilmer', 'Anthony Edwards', 'Meg Ryan']
                                            Relationship properties:
                                            The relationships:
                                                (:Actor)-[:ACTED_IN]->(:Movie)
                                        Cypher Query:
                                            MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
                                            WHERE m.name = 'Top Gun'
                                            RETURN a.name AS ActorName
                                        ) 


                                        Always consider similar types of relationships such as LIKES and PRACTICES or HOBBY and SPORT in the same query.
                                        Examples: Here are a few examples of generated Cypher statements for particular questions:
                                        
                                        # Who is interested in Climbing ? or Who likes Climbing? or Who practices Climbing?
                                        MATCH (p:Person) WHERE (p)-[:PRACTICES|LIKES|IS_INTEREST_IN]->(:Interests|Sport|Hobby {{id:"Climbing"}}) RETURN p.id AS PersonInterestedInClimbing
                                        # who likes to play chess ?
                                        MATCH (p:Person) WHERE (p)-[:PRACTICES|LIKES|IS_INTEREST_IN]->(:Interests|Sport|Hobby {{id:"Chess"}}) RETURN p.id AS PersonWhoLikesChess
                                        # who likes playing the guitar?
                                        MATCH (p:Person) WHERE (p)-[:PRACTICES|LIKES|IS_INTEREST_IN]->(:Interests|Sport|Hobby {{id:"Guitar"}}) RETURN p.id AS PersonWhoLikesGuitar
                                        # who practices swimming?
                                        MATCH (p:Person) WHERE (p)-[:PRACTICES|LIKES|IS_INTEREST_IN]->(:Interests|Sport|Hobby {{id:"Swimming"}}) RETURN p.id AS PersonWhoPracticesSwimming
                                        # how many users like to play chess, list them.
                                        MATCH (p:Person)-[:PRACTICES|LIKES|IS_INTEREST_IN]->(:Interests|Sport|Hobby {{id:"Chess"}}) RETURN count(p) AS numberOfUsers, collect(p) AS UsersWhoLikeChess
                                        # Who completed the 'Warrior' pose ? 
                                        // It does not matter if it was successful or not
                                        MATCH (p:Person)-[:COMPLETED_POSE]->(y:Yoga_pose {{id: "Warrior"}}) RETURN p.id AS PersonWhoCompletedWarriorPose
                                        # Who completed the 'Warrior' pose successfully?
                                        MATCH (p:Person)-[:COMPLETED_POSE {{success: "true"}}]->(y:Yoga_pose {{id: "Warrior"}}) RETURN p.id AS PersonWhoSuccessfullyCompletedWarriorPose
                                        # Who completed the 'Warrior' pose unsuccessfully?
                                        MATCH (p:Person)-[:COMPLETED_POSE {{success: "false"}}]->(y:Yoga_pose {{id: "Warrior"}}) RETURN p.id AS PersonWhoUnsuccessfullyCompletedWarriorPose
                                        # What is the success rate of the 'Warrior' pose?
                                        MATCH (p:Person)-[r:COMPLETED_POSE]->(y:Yoga_pose {{id: "Warrior"}})
                                        WITH 
                                            collect(CASE WHEN r.success = "true" THEN p.id END) AS successfulPersons,
                                            collect(CASE WHEN r.success = "false" THEN p.id END) AS unsuccessfulPersons,
                                            count(r) AS totalAttempts,
                                            count(CASE WHEN r.success = "true" THEN 1 END) AS successfulAttempts

                                        RETURN 
                                            successfulPersons,
                                            unsuccessfulPersons,
                                            successfulAttempts,
                                            totalAttempts,
                                            toFloat(successfulAttempts) / totalAttempts AS successRate

                                        # what are mario's interests?
                                        MATCH (p:Person {{id: "Mario"}})-[:IS_INTEREST_IN]->(target) RETURN p, target
                                        
                                        # what are mario's hobbies? or what are mario's interests?
                                        // return all mario's connections with hobbies
                                        MATCH (p:Person {{id: "Mario"}})-[r]->(h:Interests|Sport|Hobby) RETURN type(r) AS RelationshipType, h.id AS HobbyID, properties(r) AS RelationshipProperties

                                        # What are Mario's feedbacks?
                                        MATCH (p:Person {{id: "Mario"}})-[r:HAS_FEEDBACK]->(target) RETURN p, properties(r) AS FeedbackProperties, target

                                        # What are the general feedcacks?
                                        MATCH (p:Person)-[r:HAS_FEEDBACK]->(target)
                                        RETURN p, properties(r) AS FeedbackProperties, target

                                        # How often does Mario practice yoga? or How often does Mario practice yoga?
                                        MATCH (p:Person {{id: "Mario"}})-[r:PRACTICES]->(target) RETURN p, properties(r) AS PracticeProperties, target

                                        # Who is the person with the most successful poses? 
                                        // Step 1: Conta le pose completate con successo per ogni persona
                                        MATCH (p:Person)-[r:COMPLETED_POSE {{success: "true"}}]->(y:Yoga_pose)
                                        WITH p, count(r) AS successfulPoses

                                        // Step 2: Trova il massimo numero di pose completate con successo
                                        WITH collect({{person: p, successfulPoses: successfulPoses}}) AS results
                                        WITH results, reduce(maxPoses = 0, r IN results | CASE WHEN r.successfulPoses > maxPoses THEN r.successfulPoses ELSE maxPoses END) AS maxSuccessfulPoses

                                        // Step 3: Restituisci tutte le persone con il massimo numero di pose completate con successo
                                        UNWIND results AS result
                                        WITH result.person AS p, result.successfulPoses AS successfulPoses, maxSuccessfulPoses
                                        WHERE successfulPoses = maxSuccessfulPoses

                                        RETURN p.id AS PersonWithMostSuccessfulPoses, successfulPoses
                                                                  
                                        The question is:
                                        {question}"""

        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
        )


        chain = GraphCypherQAChain.from_llm(
                #self.llm,
                graph=self.graph,
                cypher_llm=self.llm,
                qa_llm=self.llm,
                verbose=True,
                validate_cypher=False, # se ci sono warning sto robo blocca la query
                #function_response_system = "Person ID is the name and surname of the people that you have to use as context as answer",
                cypher_prompt=CYPHER_GENERATION_PROMPT,
                return_intermediate_steps=True,
                allow_dangerous_requests=True,
            )
        answer = chain.invoke({"query": question})

        return answer

    def remove_aliases_from_nodes_properties(self):
        """
        Remove the 'aliases' property from all nodes in the graph.
        """
        query = """
        MATCH (n)
        REMOVE n.aliases
        """
        self.graph.query(query)


