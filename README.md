# Brain-Wiring
A group of scientists measure brain activitives from 15 brain regions of 820 subjects. Each subject is measured 4800 times for each region, and such data are stored as a 4800 x 15 matrix in a file as “subXXXXXX.txt” where XXXXXX is the subject ID. The complete dataset should contain 820 files. We will use this dataset to practice the data management and analytical tools.

Spark Version was inside _hw2_ folder

<html>
<div id="header">
<h1 class="title">Stat Learning and Big Data: Homework 2</h1>
<h2 class="author">Student: Yue Peng</h2>
</div>
<div id="TOC">
<ul>
<li><a href="#instructions-on-submitting-your-solution">Instructions on Submitting Your Solution</a></li>
<li><a href="#brain-wiring">Brain Wiring</a></li>
</ul>
</div>
<hr />
<p><strong>Instructions</strong></p>
<ul>
<li>Deadline: <strong>11:59 pm, March 10, 2018</strong> on Canvas.</li>
<li>Please start working on this early, even though the deadline is in two weeks. It is hard to predict what kind of bugs will come up!</li>
<li>You are allowed (also encouraged) to work in teams. Each team can have any number of members.</li>
<li>Each team should submit only one solution on Canvas, with all the team members clearly listed.<br /></li>
<li>The members of each team may self sign-up the project groups on Canvas. Please let me know if it does not work.</li>
<li>You are encouraged to seek help from the instructor.</li>
<li>Please submit your codes only (R or other languages). Please do not submit the data unless required by the problem.</li>
<li>You are not restricted to use R only.</li>
<li>Please submit your solution to Canvas online. Note that the online submission will be closed automatically after the deadline. Before the deadline, you may submit replacements.</li>
</ul>
<hr />
<h1 id="instructions-on-submitting-your-solution">Instructions on Submitting Your Solution</h1>
<ul>
<li>Your solution will be evaluated by <em>accuracy</em>, <em>running time</em>, and <em>code length</em> (number of characters), on a linux machine with 8g memory and 4 cores.</li>
<li>If you script requires loading libraries, please include lines to install libraries.</li>
<li>You should also include a function “<strong>authors()</strong>”, which returns a vector of names for all team members. We will use these names to record your grades!</li>
<li>Only your program is required for submission. You do not need to submit the output data or results.</li>
<li>Your program should be able to run within the folder <strong>brainD15</strong> that contains all the csv data files.</li>
</ul>
<h1 id="brain-wiring">Brain Wiring</h1>
<p>A group of scientists measure brain activitives from <strong>15</strong> brain regions of <strong>820</strong> subjects. Each subject is measured <strong>4800</strong> times for each region, and such data are stored as a <strong>4800 x 15</strong> matrix in a file as “subXXXXXX.txt” where XXXXXX is the subject ID. The complete dataset should contain 820 files. We will use this dataset to practice the data management and analytical tools.</p>
<p>The dataset file is brainD15.zip on Canvas.</p>
<ol style="list-style-type: decimal">
<li>[25%] Compute the correlation matrix (15 regions by 15 regions) for each subject, and perform Fisher’s Z transform of correlation matricces. Fisher’s Z transformed matrix <span class="LaTeX">$F$</span> is defined as <span class="LaTeX">$$
F_{ij} = \frac{1}{2} \log( \frac{1+R_{ij}}{ 1 - R_{ij}}  ) 
$$</span> where <span class="LaTeX">$R_{ij}$</span> is the correlation between Region <span class="LaTeX">$i$</span> and Region <span class="LaTeX">$j$</span>. The diagonal entries of <span class="LaTeX">$F$</span> should be set to 0 because the transformation is not defined when <span class="LaTeX">$R_{ii} = 1$</span>.
<ul>
<li>Please do not save all the files. You do not have to save all of <strong>820</strong> <span class="LaTeX">$F_s$</span> matrices, <span class="LaTeX">$s=1,..., 820$</span>. In fact, this is not possible if the data set is really large, for example with millions of subjects. Like what we discussed in class on spark, providing a function to map the data to the matrices should be sufficient.</li>
</ul></li>
<li><p>[25%] Compute the averages and variances of each entry over all the <strong>820</strong> <span class="LaTeX">$F_s$</span> matrices.</p>
<ul>
<li>Your program should write out two csv files. The first one contains the average matrix <span class="LaTeX">$F_m$</span>, and should be named as should be named as <strong>Fn.csv</strong> <em>exactly</em>. The second one named as <strong>Fv.csv</strong> contains <span class="LaTeX">$F_v$</span>.</li>
</ul></li>
<li>[25%] Order the subjects by their subject IDs from the smallest to the largest. Following problem 2, compute the average matrix <span class="LaTeX">$F_{train}$</span> for the first 410 subjects, and <span class="LaTeX">$F_{test}$</span> for the remaining 410 subjects.
<ul>
<li>Write out two csv files. The first one named as <strong>Ftrain.csv</strong> contains <span class="LaTeX">$F_{train}$</span> and the second one named as <strong>Ftest.csv</strong> contains <span class="LaTeX">$F_{test}$</span>.</li>
</ul></li>
<li>[25%] Normalize the data matrix (4800 x 15) for each subject such that each columne should have mean 0 and variance 1. Let the normalized data matrix for Subject <span class="LaTeX">$s$</span> be <span class="LaTeX">$X_s$</span>. Explore the patterns in <span class="LaTeX">$X_s$</span> and <span class="LaTeX">$F_{train}$</span>, and choose a matrix factorization method for the concatenated data <span class="LaTeX">$X_{train} \approx UG$</span> of the first 410 subjects. This matrix <span class="LaTeX">$X_{train}$</span> should have a dimension of (4800*410) x 15. Compute the covariance <span class="LaTeX">$C_{UG}$</span> of your choice of <span class="LaTeX">$UG$</span> and the sample covariance <span class="LaTeX">$C_{train}$</span> of <span class="LaTeX">$X_{train}$</span>. Which matrix is closer to the sample covariance <span class="LaTeX">$C_{test}$</span> of the next 410 subjects? The closeness between matrix <span class="LaTeX">$A$</span> and <span class="LaTeX">$B$</span> here is measured by the Frobenius norm distance <span class="LaTeX">$\| A - B \|_F$</span>.
<ul>
<li>Write out seven csv files. The first one <strong>U.csv</strong> contains <span class="LaTeX">$U$</span>, the second one <strong>G.csv</strong> contains <span class="LaTeX">$G$</span>, the third one <strong>CUG.csv</strong> contains <span class="LaTeX">$C_{UG}$</span>, the fourth one <strong>Ctrain.csv</strong> contains <span class="LaTeX">$C_{train}$</span>, the fifth one <strong>Ctest.csv</strong> contains <span class="LaTeX">$C_{test}$</span>, and the sixth one <strong>CUGCtest.csv</strong> contains <span class="LaTeX">$\| C_{UG} - C_{test} \|_F$</span>, and the seventh one <strong>CtrainCtest.csv</strong> contains <span class="LaTeX">$\| C_{train} - C_{test} \|_F$</span>. Note that your program does not need to follow the exact steps (e.g. normalization and etc) described in this problem as long as it writes out the files correctly required for this problem. The smaller <span class="LaTeX">$\| C_{UG} - C_{test} \|_F$</span>, the better your solution is for this problem.</li>
</ul></li>
<li>[Bonus 25%] You may want to try to answer problem 4 using a larger dataset with 300 regions instead of 15. The dataset (~20G) is available for download from https://www.dropbox.com/s/hdv8m7tk7rh38hk/brainD300.zip?dl=0 or <a href="https://www.dropbox.com/s/hdv8m7tk7rh38hk/brainD300.zip?dl=0">here</a>.
<ul>
<li>Write out the required seven csv files as in problem 4. Note that you will need to be careful about storing those intermediate results (espcially a huge number of large matrices). Borrowing the one-at-a-time idea (as in Spark, Files) could be helpful for calculating these matrices.</li>
</ul></li>
</ol>
</body>
</html>
