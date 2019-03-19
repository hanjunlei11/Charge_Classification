import jieba
import psycopg2
import re
def getone(texts,rules):
    pattern=re.compile(rules,re.DOTALL)
    rete=pattern.search(texts)
    if rete is not None:
        return rete.group()
    else:
        return ""
def getall(texts,rules):
    pattern=re.compile(rules)
    retes=pattern.finditer(texts)
    return retes
def getcase(zw):
    details_slcm = getone(zw, r'(|经)(|依法)审理查明\w*(|\S)(|.)(|\S\w*\S)\S*\w\S+.(|\S\w*\S)\S*\w\S+')  # .是为了匹配回车
    details_gsjg = getone(zw, r'((公诉机关|检察院)\w*指控\w*|原审认定|原判认定)(\S|)(.|)(\S\w*\S|)\S*\w\S+.(\S\w*\S|)\S*\w\S+')
    details_bg = details_slcm
    if details_bg == "" or (
            (details_bg.count("指控") or details_bg.count("原审") or details_bg.count("原判")) and details_bg.count("事实")):
        details_bg = details_gsjg
    if details_bg == "":
        details_bg = details_slcm
    return details_bg
conn = psycopg2.connect(database='justice', user='beaver', password='123456', host='58.56.137.206', port='5432')
cur=conn.cursor()

dic2count={}
length=[]

file_length=open('file_length.txt','w+')
file_words=open("file_dic.txt",'w+')


ws = open('./train.txt')
zs = ws.readlines()

for zw in zws:
    s=getcase(zw[0])
    tcs=list(jieba.cut(s))
    length.append(len(tcs))
    for word in tcs:
        if word in dic2count:
            dic2count[word]+=1
        else:
            dic2count[word]=1
    conn.commit()
vacount=dic2count.values()

#save the length of each case
length=list(map(str,length))
len_f_w=' '.join(length)
file_length.write(len_f_w)
file_length.close()

ava_word=['UNK']
ava_word_index=[0]
j=1
for i in dic2count.items():
    if i[1] >=5:
        ava_word.append(i[0])
        ava_word_index.append(j)
        j+=1
        print(j)
ava_words=' '.join(ava_word)
file_words.write(ava_words)











