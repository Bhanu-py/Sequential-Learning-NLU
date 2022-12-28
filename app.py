import streamlit as st
from annotated_text import annotated_text
from model import pretrained
from util import pre_process, predict, softmax
import torch
import numpy as np
import os

st.title("Writing skills Assessment")

st.markdown("### 🎲 Classification Application")
st.markdown("Accessing the Writing skills of a document/author by classifiying the statements and sentences into "
            "different classes based on the sequential learning using Pre-trained models from BERT. Rating the "
            "document based on the score obtained from the classes for each statement/sentence.")
menu = ["Select image from the below list", "Upload From Computer"]
choice = st.sidebar.radio(label="Menu", options=["Select .txt file from the below list", "choose your own .txt file"])
#
if choice == "Select .txt file from the below list":
    file = st.sidebar.selectbox("choose your .txt file", os.listdir("test"))
    uploaded_file = os.path.join(os.getcwd(), "test", file)
else:
    uploaded_file = st.sidebar.file_uploader("Please upload an .txt file:", type=['txt'])

# # Loading model
model = pretrained()
model.load_state_dict(torch.load('model_save/model_weights.pth'))
model.eval()


def predict(txt_file):
    with open(txt_file, 'r') as f:
        sent = [sent for sent in f.read().split("\n") if len(sent) != 0]

        print(sent)

    pred_inputs, pred_masks = pre_process(sent)

    print(pred_inputs)
    print(pred_masks)

    # logits = predict(zip(pred_inputs, pred_masks), model)
    with torch.no_grad():
        logits = model(pred_inputs, token_type_ids=None,
                       attention_mask=pred_masks)
    print(logits)
    cat = ['Claim', 'Concluding Statement', 'Counterclaim', 'Evidence', 'Lead', 'Position', 'Rebuttal']
    categories = {'Claim': 0, 'Concluding Statement': 1, 'Counterclaim': 2, 'Evidence': 3,
                  'Lead': 4, 'Position': 5, 'Rebuttal': 6}

    probs = [np.asarray(softmax(logit)) for logit in logits[0]]
    labels = np.argmax(probs, axis=1)
    print(labels)
    dic = {}
    for i, j in zip(sent, labels):
        # print(f'{i} ---- {cat[j]}')
        dic[i] = cat[j]
    return dic

#     st.image(img, caption="Uploaded image", use_column_width=True)
#
#     return pred_class_label, pred_score
# # pred_class_label, pred_score = predict('Images/12397.jpg')
# # print(pred_class_label)
#
#
if uploaded_file is not None:
    pred_dic = predict(uploaded_file)
    # st.write("**Orientation:**", pred_class_label)
    # st.write("**Probability:**", f'{round(pred_score*100)}%')
    expander = st.expander("For more details !!")
    expander.write(pred_dic)