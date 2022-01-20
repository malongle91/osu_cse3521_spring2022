---
layout: default
<script>
$('table.display').DataTable()
</script>

---

<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css" />
<script src="https://code.jquery.com/jquery-3.5.1.js"></script>
<script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>  

# Spring 2022

## Class Information

Item                     | Section 010                 
------------------------ | -----------                  
Schedule                 | Tues/Thur 2:20 PM - 3:40 PM   
 Location                | Baker Systems 180 (or ONLINE)      
 Professor               | Greg Ryslik / ryslik DOT 1 AT osu DOT edu
 Professor Office Hours  | Thursdays (3:40 - 4:40 PM) - via zoom. Contact me if you plan to attend.
 TA                      | Shuning Jiang (jiang.2126)
 TA Office Hours         | Wedensday & Friday 1:30 PM - 2:00 PM. Zoom: please check the syllabus on Carmen Canvas.



## Description: 
### Course abstract:
Survey of basic concepts and techniques in artificial intelligence, including problem-solving, knowledge representation, and machine learning.

### Course objectives :  
-Gain familiarity with basic search techniques for problem-solving.  <br>
-Gain exposure to multiple knowledge-representation formalisms. <br>
-Gain exposure to data and feature representations. <br>
-Understand basic unsupervised learning techniques and the kinds of problems they solve.  <br>
-Understand basic supervised learning techniques and the kinds of problems they solve.  <br>
-Gain expsoure to the ethics of AI.

### Course credits: 
3 units

### Note: 
It is highly encouraged that you simply fork this entire repo and then just sync updates periodically. This will provide you the easiest access to all the class materials at once. For more information on this process, you can see the documentation [here](https://docs.github.com/en/get-started/quickstart/fork-a-repo). 

## Requirements & Review Materials

### Prerequisites:

-CSE 2331 (Foundations 2) or 5331 <br>
-Linear algebra: Math 2568, 2174, 4568, or 5520H <br>
-Statistics and probability: Stat 3201 or 3450 or 3460 or 3470 or 4201 or Math 4530 or 5530H <br>
-Students in the class are expected to have a reasonable degree of mathematical sophistication and to be familiar with the basic knowledge of linear algebra, multivariate calculus, probability, and statistics. Students are also expected to have knowledge of basic algorithm design techniques and basic data structures.  <br>
-Programming in Python 3 is required.

### Review Materials:
1. [linear algebra](http://www.google.com/url?q=http%3A%2F%2Fcs229.stanford.edu%2Fsummer2020%2Fcs229-linalg.pdf&sa=D&sntz=1&usg=AFQjCNHpuHdE0Smz1gtVal60wFBolsKu2A)
2. [probability](http://www.google.com/url?q=http%3A%2F%2Fcs229.stanford.edu%2Fsummer2020%2Fcs229-prob.pdf&sa=D&sntz=1&usg=AFQjCNGAllccJuluufdQI-wjMolkXTGmnw)
3. [Python-1](http://www.google.com/url?q=http%3A%2F%2Fai.berkeley.edu%2Ftutorial.html%23PythonBasics&sa=D&sntz=1&usg=AFQjCNHmgcfln4NzCEVvCaZwn6wX4CFy4g)
4. [Python-2](https://www.google.com/url?q=https%3A%2F%2Fcs231n.github.io%2Fpython-numpy-tutorial%2F&sa=D&sntz=1&usg=AFQjCNGv73L4oqVNd6m4dBB4IKNQMZD81g)
5. [Python-3](https://www.google.com/url?q=https%3A%2F%2Fwww.python.org%2Fabout%2Fgettingstarted%2F&sa=D&sntz=1&usg=AFQjCNEenTVYVxiWp53maXVcCXDvr7fPiA)

## Textbooks:
### Required Textbooks:
-No required textbooks.

### Suggested supplemental references:
1. Stuart Russell and Peter Norvig, Artificial intelligence: a modern approach (3rd edition). Pearson, 2010 
2. Christopher M Bishop, Pattern recognition and machine learning. Springer, 2006.
3. Kevin P. Murphy, Machine Learning: A Probabilistic Perspective. The MIT Press, 2012
4. Shai Shalev-Shwartz and Shai Ben-David, Understanding machine learning: From theory to algorithms. Cambridge university press, 2014.
5. Ian Goodfellow, Yoshua Bengio, and Aaron Courville, Deep learning. MIT Press, 2016.
6. Ethem Alpaydin, Introduction to Machine Learning. The MIT Press. 
7. Kaare Brandt Petersen and Michael Syskind Pedersen, [The Matrix Cookbook](https://www.google.com/url?q=https%3A%2F%2Fwww.math.uwaterloo.ca%2F~hwolkowi%2Fmatrixcookbook.pdf&sa=D&sntz=1&usg=AFQjCNHd8gcMhQeX2KzF6f0j8dH3bGSwxw)


## Class outline

<table class="display" border=1 frame=sides rules=all>
  {% for row in site.data.Syllabus %}
    {% if forloop.first %}
    <tr>
      {% for pair in row %}
        <th>{{ pair[0] }}</th>
      {% endfor %}
    </tr>
    {% endif %}

    {% tablerow pair in row %}
      {{ 	pair[1] }}
    {% endtablerow %}
  {% endfor %}
</table>

### Miscellaneous

### Lectures
1. Due to Omicron, lectures will start off via zoom at the start of the semester. They will be recorded and posted online after each class. You are encouraged to join the zoom session live.
2. Zoom session details are posted in Carmen.

### Grading (tentative: )
- Homework (40%)
- Participation & quizzes: 10%
- Midterm exams: 25%
- Final exam: 25%

#### Important:
- Don't forget to do your participation quizzes under the assignments tab!

### Announcements & Communications:
- I will make announcements via carmen. If I need to reach out to an individual students, I will email them at name.#@osu.edu.
- Please feel free to post questions & discussions in Carmen. The TA and I will monitor them.
- If you email me a question whose answer will benefit the whole class, I might send out an anonymized version of your question & answer to the class to help everyone.

