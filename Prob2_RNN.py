# Prob2_RNN

#------------------------------ Import module---------------------------------#
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import matplotlib.pyplot as plt 

#-------------------------------- Parameter ----------------------------------#
Embedding_length=10
win=2                  # 作為相近單字的距離
sequence_length=10     # 一句話的長度
n_classes=2            # 最後結果分成2類([1,0]為accept,[0,1]為reject)
training_accuracy_record=[]
testing_accuracy_record=[]
loss_record=[]

#-------------------------------- function -----------------------------------#
def creat_dictionary(word_occur_times_dictionary,handle_paper,total_word,paper_num):
    paper_to_string=np.array_str(handle_paper)  # 轉為string
    paper_to_string=paper_to_string.lower()     # 全部轉為小寫
    
    paper_truncate=paper_to_string[2:-2]    # 不要['']
    
    # 刪掉特殊符號(非單字元素)
    paper_replace=paper_truncate.replace('\'', ',')  # 把'轉為逗號分隔
    paper_replace=paper_replace.replace(' ', ',')  # 把空格轉為逗號分隔
    paper_replace=paper_replace.replace(':', ',')  # 把:轉為逗號分隔
    paper_replace=paper_replace.replace('&', ',')  # 把&轉為逗號分隔
    paper_replace=paper_replace.replace('-', ',')  # 把-轉為逗號分隔
    paper_replace=paper_replace.replace('_', ',')  # 把_轉為逗號分隔
    paper_replace=paper_replace.replace('(', ',')  # 把(轉為逗號分隔
    paper_replace=paper_replace.replace(')', ',')  # 把)轉為逗號分隔
    paper_replace=paper_replace.replace('?', ',')  # 把?轉為逗號分隔
    
    paper_split=paper_replace.split(',')  # 依逗號拆解開來
    
    if '' in paper_split:                 # 若list包含空元素
        paper_split.remove('')            # 則刪掉他
    if ' ' in paper_split:                 # 若list包含空元素
        paper_split.remove(' ')            # 則刪掉他   
    if '  ' in paper_split:                 # 若list包含空元素
        paper_split.remove('  ')            # 則刪掉他     
         
    total_word[paper_num]=paper_split     # 存入總字典
    if '' in total_word[paper_num]:                 # 若list包含空元素
        total_word[paper_num].remove('')            # 則刪掉他   

    sentence_length=len(paper_split)      # 取得這個句子的單字數量
    
    for word_number in range(0,len(paper_split)):
        if word_occur_times_dictionary.get(paper_split[word_number],None)==None: # 沒出現過
            word_occur_times_dictionary[paper_split[word_number]]=1     # 在字典中新增
        else:
            word_occur_times_dictionary[paper_split[word_number]]+=1    # 累計加1
        if word_occur_times_dictionary.get('',None)!=None:
            del word_occur_times_dictionary['']    # 移除空字典
            
    return word_occur_times_dictionary,sentence_length,total_word

def Tokenizer(integer_tokens,paper_num,word_occur_order,total_word):
    handle_paper_string=total_word[paper_num]
    point=0
    for word_number in range(0,len(handle_paper_string)):
        if handle_paper_string[word_number] in word_occur_order:
            integer_tokens[paper_num,point]=word_occur_order.index(handle_paper_string[word_number])+1 
            point+=1
    return integer_tokens
    
# 產生Weight參數
def weight_generate(shape):
    initialize=tf.truncated_normal(shape,stddev=1/math.sqrt(float(shape[0]+shape[1])))
    return tf.Variable(initialize)

# 產生Bias參數
def bias_generate(shape):
    initialize=tf.truncated_normal(shape,stddev=1/math.sqrt(float(shape[0])))
    return tf.Variable(initialize)    
'''    
# convert numbers to one hot vectors
def to_one_hot_encoding(data_index,one_hot_dimension):
    one_hot_encoding = np.zeros(one_hot_dimension)
    one_hot_encoding[data_index] = 1
    return one_hot_encoding
'''

# 建立Embedding網路的training data
def Create_Embedding_training_data(integer_tokens,word_occur_order,win):
# 初始化矩陣
    window=np.linspace(-win,win,2*win+1)       # 相關詞的範圍
    window=np.delete(window, win)  # 刪除中間的0
    window=window.astype(int)
        
    Embedding_training_data_label=np.zeros([1,2])  # [某單字integer, 該單字附近的單字integer]
    Embedding_training_data_label[0,:]=[1,1]
    temp_training_data_label=np.zeros([2,2])
    
    # 開始建立
    for paper_num in range(0,len(integer_tokens)):  # 依序對每篇論文的題目做處理
        check_interger=0                            # 這篇論文題目的第幾個單字
        while(1):
            if check_interger>=np.size(integer_tokens,1):    # 已超過最大長度
                break;
   
            if integer_tokens[paper_num,check_interger]!=0:   # 表示這個paper_num未讀完
                for related_range in range(0,len(window)):  
                    if (check_interger+window[related_range])>=np.size(integer_tokens,1):
                        break;

                    if (check_interger+window[related_range])>=0 and integer_tokens[paper_num,(check_interger+window[related_range])]!=0:  # index不得小於0,且該位置上的值也不為0
                        temp_training_data_label[0,:]=[integer_tokens[paper_num,check_interger],integer_tokens[paper_num,(check_interger+window[related_range])]]
                        temp_training_data_label[1,:]=[integer_tokens[paper_num,(check_interger+window[related_range])],integer_tokens[paper_num,check_interger]]
               
                        Embedding_training_data_label=np.concatenate((Embedding_training_data_label,temp_training_data_label),axis=0)                       
                # 換讀下一個單字
                check_interger+=1  
            else:                                              # 表示這個paper_num已讀完
                break;                
    return Embedding_training_data_label
        
# 建立 Embedding_vector
def Create_Embedding_vector(Embedding_w,Embedding_b):
    one_hot_dimension=np.size(Embedding_w,0)
    Embedding_length=np.size(Embedding_w,1)
    Embedding_vector=np.zeros([one_hot_dimension,Embedding_length])
    for word in range(0,one_hot_dimension):     # 依序處理每個單字 
        Embedding_vector[word,:]=np.dot(np.eye(one_hot_dimension)[word,:],Embedding_w)+Embedding_b
    return Embedding_vector

#---------------------------------- Main -------------------------------------#
# 讀檔案
#WS = pd.read_excel("ICLR_accepted.xlsx")
WS = pd.read_excel("/home/alantao/deep learning/DL HW2/ICLR_accepted.xlsx")
WS_np = np.array(WS)   # 轉為矩陣
#WS = pd.read_excel("ICLR_rejected.xlsx")
WS = pd.read_excel("/home/alantao/deep learning/DL HW2/ICLR_rejected.xlsx")
WS_np =np.concatenate((WS_np,WS),axis=0) 

word_occur_times_dictionary={}  # 儲存每個單字出現的次數
total_word={}                   # 儲存每個句子中所有的單字
max_sentence_length=0           # 最長的句子長度(單字數量)

print('Create dictionary...')
# 擴充字典
for paper_num in range(0,len(WS_np)):  # 依序對每天論文的題目做處理
    handle_paper=WS_np[paper_num,:]    # 取出1列陣列(論文名稱)
    word_occur_times_dictionary,sentence_length,total_word=creat_dictionary(word_occur_times_dictionary,handle_paper,total_word,paper_num)
    
    if sentence_length>max_sentence_length:    # 若長度更長
        max_sentence_length=sentence_length    # 則更新最長長度

# 將字典轉為list,並且依照出現次數做排序(出現次數越多,排越前面)
word_occur_order=sorted(word_occur_times_dictionary, key=word_occur_times_dictionary.get, reverse=True)
#word_occur_order=word_occur_order[:int(len(word_occur_order)*3/4)]   # 後面丟掉

print('Tokenizer...')
# 將每個句子中的每個單字轉成其所對應的數字(integer)
integer_tokens=np.zeros([len(WS_np),max_sentence_length])
for paper_num in range(0,len(WS_np)):  # 依序對每天論文的題目做處理
    integer_tokens=Tokenizer(integer_tokens,paper_num,word_occur_order,total_word)

print('Create Embedding training data...')
# 建立Embedding網路的training data
Embedding_training_data_label= Create_Embedding_training_data(integer_tokens,word_occur_order,win);   

print('Starting Embedding Neural Network training...')
## Embedding Neural Network
one_hot_dimension=len(word_occur_order)  # Embedding 輸入層的維度
# input與output層
x = tf.placeholder(tf.float32, shape=(None, one_hot_dimension))
y = tf.placeholder(tf.float32, shape=(None, one_hot_dimension))
# 輸入層
W_in1=weight_generate([one_hot_dimension,Embedding_length]) 
b_in1=bias_generate([Embedding_length])
# 中間層
hidden_middle=tf.add(tf.matmul(x,W_in1), b_in1)
hidden_middle=tf.sigmoid(hidden_middle)
# 輸出層
W_out4=weight_generate([Embedding_length,one_hot_dimension])
b_out4=bias_generate([one_hot_dimension])
b_out4=tf.zeros([one_hot_dimension])
output=tf.add(tf.matmul(hidden_middle,W_out4),b_out4)
prediction = tf.nn.softmax(output)
# loss function: cross entropy
cross_entropy=-tf.reduce_sum(y*tf.log(tf.clip_by_value(prediction,1e-8,(1-(1e-8)))), axis=[1])
# training operation
train_op=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
loss = tf.reduce_mean(cross_entropy)
accuracy_judge=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))  # 輸出的機率最大者是否與label標記者相等
accuracy=tf.reduce_mean(tf.cast(accuracy_judge,'float'))  # 轉為float並且做平均(多筆data)
# 啟動網路
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# training parameter
embedding_epoch_num=3         # epoch 5次
embedding_batch_size=50
embedding_data_number=len(Embedding_training_data_label)
# 開始訓練
for epoch_times in range(0,embedding_epoch_num):
    print('epoch times=',epoch_times)
    for batch_times in range(0,int(embedding_data_number/embedding_batch_size)):  # 全部的資料可以分成多少個batch
        temp_data=Embedding_training_data_label[batch_times*embedding_batch_size:(batch_times+1)*embedding_batch_size,0]     # 獲得training data
        temp_data-=1     # 因為index應該從0開始
        get_x=np.eye(one_hot_dimension)[temp_data.astype(int),:]      # training data轉為one-hot vector           
        temp_label=Embedding_training_data_label[batch_times*embedding_batch_size:(batch_times+1)*embedding_batch_size,1]     # 獲得training label
        temp_label-=1     # 因為index應該從0開始
        get_y=np.eye(one_hot_dimension)[temp_label.astype(int),:]     # training label轉為one-hot vector
        sess.run(train_op, feed_dict={x: get_x, y: get_y})
        
    # 計算正確率(每個epoch都會算一次training正確率)
    temp_data=Embedding_training_data_label[:,0]     # 獲得training data
    temp_data-=1     # 因為index應該從0開始
    get_x=np.eye(one_hot_dimension)[temp_data.astype(int),:]      # training data轉為one-hot vector
    temp_label=Embedding_training_data_label[:,1]     # 獲得training label
    temp_label-=1     # 因為index應該從0開始
    get_y=np.eye(one_hot_dimension)[temp_label.astype(int),:]     # training label轉為one-hot vector
    training_accuracy=sess.run(accuracy,feed_dict={x: get_x, y: get_y})
    print('training_accuracy=',training_accuracy)
        
    # 每做完1次epoch就做shuffle
    np.random.shuffle(Embedding_training_data_label)     # shuffle
print('Embedding Neural Network training complete...')

# 取出Embedding矩陣
Embedding_w = sess.run(W_in1)
Embedding_b = sess.run(b_in1)

# clear computational graph
sess.close()
tf.reset_default_graph()

print('Change Word to Embedding vector...')
# 存放不同單字所代表的Embedding vector
Embedding_vector=Create_Embedding_vector(Embedding_w,Embedding_b)

print('Change Sequence to Embedding vector...')
# 存放每句話的Embedding vector
Embedding_sequence=np.zeros([np.size(integer_tokens,0),Embedding_length,sequence_length])
for paper_num in range(0,np.size(integer_tokens,0)):
    for word in range(sequence_length):
        if integer_tokens[paper_num,word] !=0:   #該處有字
            Embedding_sequence[paper_num,:,word]=Embedding_vector[int(integer_tokens[paper_num,word]-1),:]
            
# 切割出training data與testing data, 並且建立label
Testing_Embedding_sequence=Embedding_sequence[0:50,:,:]
Training_Embedding_sequence=Embedding_sequence[50:581,:,:]  
Testing_Embedding_sequence=np.concatenate((Testing_Embedding_sequence,Embedding_sequence[582:632,:,:]),axis=0)    
Training_Embedding_sequence=np.concatenate((Training_Embedding_sequence,Embedding_sequence[632:,:,:]),axis=0)                    

Testing_data_index=np.zeros([np.size(Testing_Embedding_sequence,0),3])  # 儲存index與對應label    [index,0,1]或[index,1,0]
Training_data_index=np.zeros([np.size(Training_Embedding_sequence,0),3])  # 儲存index與對應label
Testing_data_index[:,0]=np.linspace(0,np.size(Testing_Embedding_sequence,0)-1,np.size(Testing_Embedding_sequence,0))   # data的index先按順序
Testing_data_index[0:50,1]=1
Testing_data_index[50:,2]=1
Training_data_index[:,0]=np.linspace(0,np.size(Training_Embedding_sequence,0)-1,np.size(Training_Embedding_sequence,0))   # data的index先按順序
Training_data_index[0:532,1]=1
Training_data_index[532:,2]=1

print('Data preprocess complete!! Starting to training RNN')
     
## RNN
sess2=tf.InteractiveSession() 
#sess2 = tf.Session()
sess2.run(tf.global_variables_initializer())

layer1_length=500
layer2_length=100

batch_size=50
# input與output層
x1=tf.placeholder(tf.float32,[None,sequence_length,Embedding_length]) # input data,每個batch的所有sequence
y=tf.placeholder(tf.float32,[None,n_classes])
X=tf.reshape(x1,[-1,Embedding_length]) # [batch_size*sequence_length,Embedding_length]

# 第1層Layer
w_in2=weight_generate([Embedding_length,layer1_length])
b_in2=bias_generate([layer1_length])
X_in=tf.matmul(X,w_in2)+b_in2
X_in=tf.reshape(X_in,[-1,sequence_length,layer1_length]) # [batch_size,sequence_length,layer1_length]

# BasicRNNCell

#basic_cell=tf.contrib.rnn.BasicRNNCell(num_units=layer1_length,activation=tf.nn.relu)
#output,states=tf.nn.dynamic_rnn(basic_cell,X_in,dtype=tf.float32)

# BasicLSTMCell
lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=layer1_length,forget_bias=1.0,state_is_tuple=True)
batchsize = tf.shape(x1)[0]   # 取得batch size
initial=lstm_cell.zero_state(batch_size=batchsize,dtype=tf.float32)
output,states=tf.nn.dynamic_rnn(lstm_cell,X_in,dtype=tf.float32, initial_state=initial,time_major=False)

# MultiRNNCell
#def get_a_cell(output_size):
#    return tf.nn.rnn_cell.BasicRNNCell(num_units=layer1_length,activation=tf.nn.relu)
#cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(layer1_length) for _ in range(2)])
#batchsize = tf.shape(x1)[0]   # 取得batch size
#h0 = cell.zero_state(batch_size=batchsize, dtype=tf.float32)  # 初始狀態
#output, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=h0, time_major=False)


Output=tf.unstack(tf.transpose(output,[1,0,2])) # 每個時序的output展開

# Dropout
Output = tf.nn.dropout(Output, rate=0.1)  # 有0.1的機率會丟掉輸入

# 第3層Layer
w_out1=weight_generate([layer1_length,n_classes])
b_out1=bias_generate([n_classes])
hidden3=tf.matmul(Output[-1],w_out1)+b_out1

# RNN output
predict=tf.nn.softmax(hidden3)
#predict=tf.clip_by_value(predict,1e-10,(1-(1e-10)))

# loss function: cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
#cross_entropy=-tf.reduce_sum(y*tf.log(tf.clip_by_value(predict,1e-10,(1-(1e-10)))), axis=[1])
#cross_entropy1=-tf.reduce_sum(y*tf.log(predict), axis=[1])
#Loss = tf.reduce_mean(cross_entropy1)

# training operation(fix learning rate)
#train_op2=tf.train.AdamOptimizer(1e-5).minimize(cost)
train_op2=tf.train.AdamOptimizer(1e-4).minimize(cost)

# learning rate decay
# [decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)]
#initial_learning_rate = 1e-4                    # 初始的learning rate
#global_step = tf.Variable(0, trainable=False)   # 初始時迭代0次
#learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step=global_step,decay_steps=500,decay_rate=0.9) # 每500epoch，學習綠會衰減0.9
#train_op2=tf.train.AdamOptimizer(learning_rate).minimize(cost)



#正確率計算
accuracy_judge=tf.equal(tf.argmax(predict,1),tf.argmax(y,1))  # 輸出的機率最大者是否與label標記者相等
accuracy=tf.reduce_mean(tf.cast(accuracy_judge,'float'))  # 轉為float並且做平均(多筆data)


# loss function: cross entropy
#Loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=y))
# training operation
#train_op=tf.train.AdamOptimizer(1e-2).minimize(Loss)
#accuracy_judge=tf.equal(tf.argmax(predict,1),tf.argmax(y,1))
#accuracy=tf.reduce_mean(tf.cast( accuracy_judge ,'float'))

# 啟動網路
sess2.run(tf.global_variables_initializer())
# training parameter
epoch_num=3000         # epoch 100次

data_number=len(Training_data_index)
# 開始訓練

for epoch_times in range(0,epoch_num):
    print('epoch times=',epoch_times)
    for batch_times in range(0,int(data_number/batch_size)):  # 全部的資料可以分成多少個batch
        temp_data=Training_data_index[batch_times*batch_size:(batch_times+1)*batch_size,0]     # 獲得training data index
        get_x=Training_Embedding_sequence[temp_data.astype(int),:,:]      # 取得1個batch_size的training data  
        get_x=get_x.swapaxes(1,2)  # 把sequence_length與Embedding_length維度交換
        get_y=Training_data_index[batch_times*batch_size:(batch_times+1)*batch_size,1:3]     # 獲得training label      
        #---------------------------
        #sess2.run(learning_rate,feed_dict={global_step: epoch_times})  # 修正learning rate
        #-------------------------
        sess2.run(train_op2, feed_dict={x1: get_x, y: get_y})
               
    # Training Accuracy (每個epoch都會算一次正確率)
    temp_data=Training_data_index[:,0]     # 獲得所有training data的index
    get_x=Training_Embedding_sequence[temp_data.astype(int),:,:]   
    get_x=get_x.swapaxes(1,2)  # 把sequence_length與Embedding_length維度交換
    get_y=Training_data_index[:,1:3]     # 獲得training label
    training_accuracy=sess2.run(accuracy,feed_dict={x1: get_x, y: get_y})
    training_accuracy_record.append(training_accuracy)
    # 計算Training Loss
    Loss=sess2.run(cost,feed_dict={x1: get_x, y: get_y})
    loss_record.append(Loss)
    
    
    # Testing Accuracy (每個epoch都會算一次正確率)
    temp_data=Testing_data_index[:,0]     # 獲得所有training data的index
    get_x=Testing_Embedding_sequence[temp_data.astype(int),:,:]   
    get_x=get_x.swapaxes(1,2)  # 把sequence_length與Embedding_length維度交換
    get_y=Testing_data_index[:,1:3]     # 獲得training label
    testing_accuracy=sess2.run(accuracy,feed_dict={x1: get_x, y: get_y})
    testing_accuracy_record.append(testing_accuracy)
           
    print('Training accuracy=',training_accuracy,', Testing accuracy=',testing_accuracy,' ,Loss=',Loss)
            
    # 每做完1次epoch就做shuffle
    np.random.shuffle(Training_data_index)     # shuffle
    
# Print出結果
plt.figure(3)
plt.plot(training_accuracy_record)
plt.plot(testing_accuracy_record)
plt.xlabel('Number of epoch')
plt.ylabel('Accuracy')
plt.show() 

plt.figure(4)
plt.plot(loss_record)
plt.xlabel('Number of epoch')
plt.ylabel('Cross entropy')
plt.show()     
   

