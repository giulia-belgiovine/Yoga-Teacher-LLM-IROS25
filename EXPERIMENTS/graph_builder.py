"""
This script is used to build the graph based on all the interactions in the memory folder (summary.txt files)
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.llmYogaTeacher.knowledge_Graph.knowledge_graph import KnowledgeGraph


kg = KnowledgeGraph()

MEMORY_FOLDER = "/usr/local/src/robot/cognitiveInteraction/yoga-teacher-llm/EXPERIMENTS/yoga_tutor_memories"

for profile in os.listdir(MEMORY_FOLDER):
    for interaction in os.listdir(os.path.join(MEMORY_FOLDER, profile)):
        summary_path = os.path.join(MEMORY_FOLDER, profile, interaction, "Summary_new.txt")
        print(summary_path)
        kg.build_knowledge_graph(summary_path, None)
