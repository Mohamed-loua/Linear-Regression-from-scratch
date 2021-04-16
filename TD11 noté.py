#!/usr/bin/env python
# coding: utf-8

# # TD11 noté du 15/04

# In[46]:


from data_extraction import *
from GradDesc import *


# In[14]:


X[1234]


# In[15]:


Z[1234]


# In[16]:


Y[1234]


# In[26]:


print(X[0])


# In[23]:


plt.imshow(Z[0])


# In[27]:


print(Y[0])


# In[22]:


X.shape


# In[29]:


Y_list= Y.tolist()


# In[33]:


Y_list.count(17)


# In[35]:


S = np.column_stack((X,Y))


# In[36]:


def split_train_test(S, size = 0.1):
    S_learn = []
    S_test = []
    for s in S:
        if np.random.rand()< size:
            S_test.append(s)
        else:
            S_learn.append(s)
    return np.array(S_learn), np.array(S_test)


# In[37]:


train_set,test_set =  split_train_test(S)


# In[45]:


def One_hot_encoding(y):
    y_temp = np.zeros((y.shape[0],26))
    for i in range(y.shape[0]):
        y_temp[i, int(y[i])] =  1
        
    return y_temp

#Test: o est la première lettre dans le dataset, c'est la 15ème lettre de l'alphabet donc elle est bien placée 
print(One_hot_encoding(Y)[0])
print((One_hot_encoding(Y)).shape)


# In[47]:


def sigm(v):
    v2 = np.exp(-v)+1
    v2 = 1/v2
    return(v2)


# In[92]:


# fonction pour compter le nbr d'erreur
def nbr_error(u,S):
    In = S[:,:-1]
    Out = S[:,-1]
    
    u = u.reshape(26,129)
    x = u[:,:-1]
    b = u[:,-1]
    
    vect = In.dot(x.T) + b
    vect = softmax(vect)
    
    
    y_prediction =  np.array(vect.argmax(axis=1))
    
    
    return (Out != y_prediction).sum()

#fonction de calcul de la loss
def loss(u,S):
    In = S[:,:-1]
    Out = S[:,-1]
    
    u = u.reshape(26,129)
    x = u[:,:-1]
    b = u[:,-1]
    
    vect = In.dot(x.T) + b
    vect = sigm(vect)
    y = One_hot_encoding(Out)
    y_temp =  (y * vect).sum(axis=1)
    
    loss = -np.log(y_temp).sum()
    return loss


# In[49]:


print(X.shape)


# In[62]:


def softmax(v):
    return(((np.exp(v).T)/(np.exp(v).sum(axis = 1))).T)


# In[93]:


u = np.random.randn(3354)
print(loss(u,S))
print(nbr_error(u,S))


# In[ ]:


grad = grad_desc_n(loss, S, 3354, 1, step = 0.5, x_0 = None)

u_solution = grad

print(loss(u_solution,S))
print(nbr_error(u_solution,S))


# In[ ]:




