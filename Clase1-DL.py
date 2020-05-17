
# coding: utf-8

# # Lesson 1 - What's your Camelidae

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# We import all the necessary packages. We are going to work with the [fastai V1 library](http://www.fast.ai/2018/10/02/fastai-ai/) which sits on top of [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163). The fastai library provides many useful functions that enable us to quickly and easily build neural networks and train our models.

# In[2]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[3]:


bs = 64


# ## Looking at the data

# path a las datos:

# In[4]:


path = '/notebooks/course-v3/nbs/dl1/animales/'


# The first thing we do when we approach a problem is to take a look at the data. We _always_ need to understand very well what the problem is and what the data looks like before we can figure out how to solve it. Taking a look at the data means understanding how the data directories are structured, what the labels are and what some sample images look like.
# 
# The main difference between the handling of image classification datasets is the way labels are stored. In this particular dataset, labels are stored in the filenames themselves. We will need to extract them to be able to classify the images into the correct categories. Fortunately, the fastai library has a handy function made exactly for this, `ImageDataBunch.from_name_re` gets the labels from the filenames using a [regular expression](https://docs.python.org/3.6/library/re.html).

# En este caso el path a las imagenes es el mismo que a los datos xq no tengo más datos que las imagenes.
# Vemos como estan guardados los nombres de las imagenes para poder hacer bien los re.

# In[5]:


fnames = get_image_files(path)
fnames[:5]


# In[6]:


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# ## Training: resnet50 

# Basically, resnet50 usually performs better than resnet34 because it is a deeper network with more parameters. To help it along, let's us use larger images too, since that way the network can see more detail. We reduce the batch size a bit since otherwise this larger network will require more GPU memory.

# Vamos a juntar las imagenes en data, recortarlas con size de ser necesario para que sean todas iguales
# y normalizarlas para por ejemplo setear el brillo de todas de la misma manera etc.

# In[7]:


data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2).normalize(imagenet_stats)


# Vemos algunas de las imagenes:

# In[8]:


data.show_batch(rows=3, figsize=(7,6))


# Con esta linea aprende: (metrics=accurancy me devuelve todo en base a la eficacia, en cambio metrics=error_rate me devuelve en base al error)

# In[9]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy)


# Como sería el aprendizaje si no tomo en cuenta la curva:

# In[10]:


learn.fit_one_cycle(5)


# busco el min optimo:(yo quiero que loss sea chico, pero no me sirve que sea el minimo de la curva)
# 1)veo como es dicha curva

# In[10]:


learn.lr_find()
learn.recorder.plot()


# Pido que me diga cual es el valor optimo graficamente(min numerical gradient)

# In[11]:


learn.recorder.plot(suggestion=True)


# Le aplico dicho min:

# In[12]:


min_grad_lr = learn.recorder.min_grad_lr
min_grad_lr
print (min_grad_lr)
learn.fit_one_cycle(5, min_grad_lr)


# In[13]:


interp = ClassificationInterpretation.from_learner(learn)


# In[14]:


interp.most_confused()


# In[15]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[16]:


interp.plot_top_losses(9, figsize=(15,11))


# In[17]:


learn.save('stage-1-50')

