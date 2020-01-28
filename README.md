# Event Detection

In Natural Language Processing, an event denotes a change in states and is usually described by a set of participants(i.e. attributes, roles) doing actions. With each event comes an event trigger(s) which can be seen as what invokes the event. These triggers more than often can be seen in the form of verbs and nouns, while sometimes can be expressed through adverbs and pronouns. 

Identifying event triggers and classifying them into their respective categories are called Event Detection. It's a challenging subtask of Event Extraction since 1 event can have multiple forms of event triggers, and vice versa, the same event trigger might lead to different events depending on the context.

### DAniEL (Data Analysis for Information Extraction in any Languages): event-detection-daniel

In the context of the NewsEye project where many languages are considered, the DAniEL system, created by Gaël Lejeune, was chosen. The system focuses on Epidemic Surveillance over press articles across multiple languages. Rather depending on language-specific grammars and analyzers, the system implements a string-based algorithm that detects repeated maximal substrings in salient zones to extract important information (disease name, location). The decision of choosing prominent text zones is based on the characteristics of a journalistic writing style where crucial information is usually put at the beginning and end of the article.

This is the Python 3 version of the Daniel System (originally from here: https://github.com/rundimeco/daniel)

### Convolutional Neural Network model for event detection: event-detection-pytorch


