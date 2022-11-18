from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from conf import Config


def run_summarizer(text):
    if os.path.exists(text):
        # Open and read the article
        f = open(text, "r", encoding="utf8")
        text = f.read()

    # load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(Config.PREDICT['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(Config.PREDICT['tokenizer_name'])

    # generate output
    tokens_input = tokenizer.encode("summarize: "+ text,
                                    return_tensors='pt',
                                    max_length=Config.PREDICT['encode_max_length'],
                                    truncation=Config.PREDICT['encode_truncation'])
    ids = model.generate(tokens_input,
                         min_length=Config.PREDICT['generate_min_length'],
                         max_length=Config.PREDICT['generate_max_length'])
    summary = tokenizer.decode(ids[0], skip_special_tokens=True)
    return summary

if __name__ == '__main__':

    text_example = '''
        The tower is 324 meters (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, 
        measuring 125 meters (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure 
        in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 meters. 
        Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 meters (17 ft). Excluding transmitters, 
        the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.
    '''
    print(run_summarizer(text_example))
    
    file_path = './data/sample_news.txt'
    print(run_summarizer(file_path))

    
