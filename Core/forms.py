from django import forms


Choices=[
    ('0','sadness'),
    ('1','anger'),
    ('2','love'),
    ('3','surprise'),
    ('4','fear'),
    ('5','joy'),
]

class PredictForm(forms.Form):
    query= forms.CharField(label_suffix=' ',label="Ask Your Query",widget=forms.Textarea)


class AddDataForm(forms.Form):
    emotion = forms.ChoiceField(choices=Choices)
    type = forms.CharField(label_suffix=' ',label="Write Emtion-Text",widget=forms.Textarea(attrs= {'cols': 50, 'rows': 5}))

