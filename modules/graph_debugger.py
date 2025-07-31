import os
from llmYogaTeacher.knowledge_Graph.knowledge_graph import KnowledgeGraph


knowledge_graph = KnowledgeGraph()
ROOT_FOLDER = "/usr/local/src/robot/cognitiveInteraction/yoga-teacher-llm/fake summaries"

# for f in os.listdir(ROOT_FOLDER):
#     if f.endswith(".txt"):
#         knowledge_graph.build_graph_from_file(os.path.join(ROOT_FOLDER, f))


# knowledge_graph.build_graph_from_file("/usr/local/src/robot/cognitiveInteraction/yoga-teacher-llm/fake summaries/Elena Kirov.txt")

print(knowledge_graph.ask_graph("How did sophia found the poses?"))

# knowledge_graph.get_personal_graph("Elena Kirov","how many poses did Elena Kirov execute?")