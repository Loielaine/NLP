import re
def SearchEmail(line):
    pattern= [r'[^\<\>@ ]+@[A-z0-9]+\.[A-z]{3}',
    r'([^\<\> ]+) [\[\/]?at[\]\/]? ([A-z0-9]+) [\[\/]?dot[\]\/]? ([A-z]{3})',
    r'([^\<\> ]+) [\[\/]?@[\]\/]? ([A-z0-9]+) [\[\/]?\.[\]\/]? ([A-z]{3})',
    r'([^\<\> ]+)@([A-z0-9]+)\.([A-z]{3})']
    i = 0
    result = re.search(pattern[0],line)
    while(result is None and i<=3):
        result = re.search(pattern[i],line)
        i +=1
    return result

def ReplaceEmail(line):
    line = line.strip().replace(" at ","@").replace(" dot ",".").replace(' ','')
    line = line.replace("/at/", "@").replace("[at]","@").replace("@@","@")
    line = line.replace("/dot/", ".").replace("[dot]",".").replace("..",'dot.')
    return line

with open("W20_webpages.txt","r") as text, open('email-outputs.csv', 'w') as output:
    output.write('ID,Category\n')
    cnt = 0
    for line in text:
        result = SearchEmail(line)
        output.write(str(cnt)+',')
        cnt += 1
        if result is None:
            output.write('None\n')
        else: 
            email = ReplaceEmail(result.group())
            output.write(email+'\n')
       
       