from transformers import BertTokenizer, BertModel
from bert_score import score

# Import the necessary libraries
from bert_score import BERTScorer

# Initialize the BERTScorer object
scorer = BERTScorer(lang='en')

# Define the reference and candidate sentences
reference = ["Defined as a field at the intersection of computer science and linguistics, it is concerned with the computational aspects of the human language faculty. It is closely related to NLP but is considered more theoretical."]
candidate = ["Computational Linguistics, also known as natural language processing, is the interdisciplinary field that focuses on the use of computers to process and analyze natural language, such as speech and text. It involves the application of linguistic theories and methods to develop technologies that enable interactions between humans and machines using language. This field is crucial for the development of speech and language technologies, including voice recognition, machine translation, and language understanding in chatbots."]

# Compute the BERTScore
P, R, F1 = scorer.score(candidate, reference)

# Print the scores
print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(P.item(), R.item(), F1.item()))