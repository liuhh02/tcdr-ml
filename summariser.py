from Bio import Entrez, Medline
from io import StringIO
import pandas as pd

def getsummary(allAnswersTxt, SUMMARY_TOKENIZER, SUMMARY_MODEL, torch_device):
    answers_input_ids = SUMMARY_TOKENIZER.batch_encode_plus([allAnswersTxt], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
    summary_ids = SUMMARY_MODEL.generate(answers_input_ids,
                                               num_beams=10,
                                               length_penalty=1.2,
                                               max_length=1024,
                                               min_length=64,
                                               no_repeat_ngram_size=4)
    exec_sum = SUMMARY_TOKENIZER.decode(summary_ids.squeeze(), skip_special_tokens=True)
    return exec_sum

def getrecord(id, db):
    handle = Entrez.efetch(db=db, id=id, rettype='Medline', retmode='text')
    rec = handle.read()
    handle.close()
    return rec

def pubMedSearch(terms, db='pubmed', mindate='2019/12/01'):
    handle = Entrez.esearch(db = db, term = terms, retmax=10, mindate=mindate)
    record = Entrez.read(handle)
    record_db = {}
    for id in record['IdList']:
        try:
            record = getrecord(id,db)
            recfile = StringIO(record)
            rec = Medline.read(recfile)
            if 'AB' in rec and 'AU' in rec and 'LID' in rec and 'TI' in rec:
                if '10.' in rec['LID'] and ' [doi]' in rec['LID']:
                    record_db['pm_'+id] = {}
                    record_db['pm_'+id]['authors'] = ' '.join(rec['AU'])
                    record_db['pm_'+id]['doi'] = '10.'+rec['LID'].split('10.')[1].split(' [doi]')[0]
                    record_db['pm_'+id]['abstract'] = rec['AB']
                    record_db['pm_'+id]['title'] = rec['TI']
        except:
            print("Problem trying to retrieve: " + str(id))
        
    return record_db
Entrez.email = 'pubmedemail@gmail.com'

def getLink(link):
    link = "http://dx.doi.org/" + link
    return link

results = pubMedSearch("Covid19 Vaccine")
results = pd.DataFrame(results).transpose()

results['summary'] = results['abstract'].apply(getsummary)
results['link'] = results['doi'].apply(getLink)
results = results[['title', 'link', 'summary']]
results_dict = results.to_dict('index')
