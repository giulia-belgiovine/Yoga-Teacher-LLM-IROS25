import os
import json

memories_path = "/usr/local/src/robot/cognitiveInteraction/yoga-teacher-llm/EXPERIMENTS/yoga_tutor_memories/"

def generate_questions(user_name):
    """Generate a list of questions replacing user with the actual user name."""
    base_questions = [
        "Which sports does {user} practice?",
        "What are {user} interests?",
        "How often does {user} practice sport?",
        "What is the training level for {user} ?",
        "How many poses did {user} complete successfully?",
        "Did {user} provide some comments or feedback about the training?"
    ]
    return [q.format(user=user_name) for q in base_questions]

def find_summary_file(user_path):
    """Find the summary.txt file inside an interaction folder."""
    for root, dirs, files in os.walk(user_path):
        if os.path.basename(root).startswith("interaction") and "Summary.txt" in files:
            return os.path.join(root, "Summary_new.txt")
    return None

def read_summary(summary_path):
    """Read the content of the summary.txt file."""
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""

# 1) Create JSON of factual queries
def create_factual_json(path):
    """Iterate over user folders and create a JSON structure."""
    users_data = []

    for user_name in os.listdir(path):
        user_path = os.path.join(path, user_name)
        if os.path.isdir(user_path):
            summary_file = find_summary_file(user_path)
            groundtruth = read_summary(summary_file)

            questions = generate_questions(user_name)

            user_entry = {
                user_name: {
                    "GroundTruth": groundtruth,
                    "Questions": questions
                }
            }
            users_data.append(user_entry)

    final_data = {"Category": "Factual", "Users": users_data}

    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yoga_factual_queries.json")

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(final_data, json_file, indent=4, ensure_ascii=False)

    print(f"JSON file created successfully: {json_file_path}")


# 2) Create JSON of general (graph-wise) queries
# Eventually to add
# "Who completed the albero pose successfully?",
# "Who completed the guerriero pose successfully?",
# "Who completed the aeroplano pose successfully?",
# "Who completed the granchio pose successfully?",
# "Who completed the albero pose successfully?",
# "Who completed the fenicottero pose successfully?",
# "Who completed the lottatore pose successfully?",
# "Who completed the cactus pose successfully?",
# "Who completed the ruota pose?",
# "Who completed the triangolo pose",

general_questions = [
        "How many people are there in total?",
        "How many people currently practice swimming? List them.",
        "How many people practice, like or have an interest in hiking? List them.",
        "How many people practices, like or have an interest in photography? List them.",
        "How many people practices, like or have an interest in chess? List them.",
        "How many people practices, like or have an interest in cooking? List them.",
        "How many people practices, like or have an interest in bouldering and rock climbing? List them.",
        "How many people practices, like or have an interest in gardening? List them.",
        "How many people  practices, like or have an interest in pottery? List them.",
        "How many people practices, like or have an interest in painting? List them.",
        "Are people interested in pottery also interested in painting? Show an example.",
        "How many people have a beginner training level?",
        "Who are people with a beginner training level?",
        "How many people have an intermediate training level?",
        "Who are people with an intermediate training level?",
        "How many people have an advanced training level?",
        "Who are people with an advanced training level?",
        "How many people have completed 0 poses successfully?",
        "How many people have completed 1 pose successfully? List their name (followed by poses done successfully in brackets) ",
        "How many people have completed 2 poses successfully? List their name (followed by poses done successfully in brackets)",
        "How many people have completed 3 poses successfully?",
        "Are there more beginners, intermediate or advanced people?",
        "Are there more recurrent sports practiced by users? List the top 3 most recurrent sports",
        "How many people are there who do not practice any sport? List them",
        "Do people who never practice sport always fail in completing the proposed poses? Show the examples.",
        "Are there more people who practice sports regularly, occasionally or never?",
        "What is the success rate for beginners in total (compute it as all poses successfully completed by beginners and the total poses made by beginners)?",
        "What is the failure rate for beginners (compute it as all poses failed by beginners and the total poses made by beginners)?",
        "Are there poses which are more likely to be failed by beginners (compute it as the total number in which the pose has been failed by beginners divided by the total number in which that pose has been done. Express it in percentage)",
        "What is the success rate for intermediate in total (compute it as all poses successfully completed by intermediate and the total poses made by intermediates)?",
        "What is the failure rate for intermediate (compute it as all poses failed by intermediate and the total poses made by intermediate)?",
        "Are there poses which are more likely to be failed by intermediates? (compute it as the total number in which one pose has been failed by intermediates divided by the total number in which that pose has been done. Express it in percentage)",
    ]
groundtruths = [
    "there are 28 people in total",
    "Only one user actually practice swimming: Liam Patel",
    "There are 5 people that practice hiking: Mark Evans, Olivia Torres, Laura Peterson, Daniel Carter, Ethan Brooks",
    "There are 4 people that practice photography: Alex Novak, Chris Hall, Laura Peterson, David Chen",
    "There are 3 people that practice chess: John Martinez, David Chen, Chris Hall",
    "There are 2 people that practice cooking: John Martinez, Olivia Torres",
    "There are 2 people that practice bouldering and rock climbing: Taylor Reed, James Lee",
    "There are 3 people that practice gardening: David Chen, Margaret Hughes, Emily Carter",
    "There are 2 people that do pottery: Emily Carter, Olivia Simmons",
    "There are 3 people that practice painting: Emily Carter, Olivia Simmons, Maria Russo",
    "Yes. As an example Emily Carter and Olivia Simmons share the same interest for pottery and painting",
    "There are 11 people with a beginner training level",
    "The people with a beginner training level are:\n\n1. Mark Evans\n2. Nancy Adams\n3. Jordan Ellis\n4. Michael Johnson\n5. David Chen\n6. Ethan Brooks\n7. Sarah Quinn\n8. Mia Harper\n9. Maria Russo\n10. Lily Parker\n11. Lisa Wright",
    "There are 15 people with an intermediate training level",
    "The people with an intermediate training level are:\n\n1. John Martinez\n2. Sophie Zhang\n3. Alex Novak\n4. Taylor Reed\n5. James Lee\n6. Olivia Torres\n7. Margaret Hughes\n8. Helen Clarke\n9. Liam Patel\n10. Ryan Brooks\n11. Laura Peterson\n12. Emily Carter\n13. Chris Hall\n14. Olivia Simmons\n15. Daniel Carter",
    "There are 2 people with an advanced training level",
    "People with an advanced training level are Elena Gomez and Hannah Bell",
    "Zero people have completed 0 poses successfully",
    "9 people have completed 1 pose successfully: Daniel Carter (lottatore), Helen Clarke (albero), John Martinez (fenicottero), liam patel (fenicottero), lily parker (Aeroplano), lisa wriht (Granchio), Mark Evans (aeroplano), Olivia Simmons (Fenicottero), Taylor Reed (Fenicottero).",
    "11 people have completed 2 poses successfully: Sophie Zhang (fenicottero, albero), Sarahn Quinn (aeroplano, guerriero), Ryan Brooks (fenicottero', 'lottatore), Michael Johnson (granchio, guerriero), Margaret Hughes ('albero', 'fenicottero'), Laura Peterson (albero, lottatore), Jordan Ellis ('aeroplano', 'guerriero'), James Lee (albero, lottatore), Hannah Bell (triangolo, ruota), Emily Carter (albero, lottatore), David Chen (areoplano, granchio)",
    "8 people have completed 3 poses successfully",
    "There are more intermediate people than beginners or advanced ones.",
    "Most recurrent sports are Hiking, Yoga and walking",
    "2 people do not practice any sport: Maria Russo and Michael Jonson",
    "No because Maria did successfully all the 3 poses and Michael did successfully 2 poses out of 3",
    "There are more people practicing sports regularly",
    "Success Rate for all beginners is approximately 70%",
    "Failure rate for all beginners is 30%",
    "Guerriero and granchio pose have highest failure rate of 36% (4 out of 11)",
    "Success Rate for all intermediate is 60%",
    "Failure rate for all intermediate is 40%",
    "Lottatore have highest failure rate of 46% (7 fails out of 15)",
]


# now we can create the json file with this structure
# {
#     "Category": "General",
#     "Questions": [{
#         "Question": "Are individuals who practice sports constantly more likely to succeed in yoga than those who practice sports occasionally?\n?",
#         "GroundTruth": ""
#     }]
# }

def create_general_json(questions, groundtruths):
    """Create a JSON file with general questions."""
    general_data = {
        "Category": "General",
        "Questions": [{"Question": q, "GroundTruth": g} for q,g in zip(questions, groundtruths)]
    }

    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yoga_general_queries.json")

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(general_data, json_file, indent=4, ensure_ascii=False)

    print(f"JSON file created successfully: {json_file_path}")

# create_general_json(questions=general_questions, groundtruths=groundtruths)
create_factual_json("/usr/local/src/robot/cognitiveInteraction/yoga-teacher-llm/EXPERIMENTS/yoga_tutor_memories")