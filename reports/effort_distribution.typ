= Team effort distribution report

#show link: set text(fill: blue)

*Assignment number:* 3

*Assignment team number:* 12

*GitHub URL:* #link("https://github.com/znasser60/scicomp-assignment-3")

As with the previous assignment, we met prior to starting the third assignment, to 
discuss the parts of the previous project that went well, those that didn't, and 
how how improve the experience in this project. We then scheduled an in-person 
meeting to discuss the theoretical aspects of this assigment and lay the 
foundations for the code. 

We then divided the tasks. Zainab implemented the eigenmode problem solution for the 
circular domain while Marcell worked on the square and rectangular domains. Henry 
distilled the theory for the report, and then implemented the harmonic 
oscillator simulation. Zainab and Henry met to discuss the treatment of sources and 
sinks in the steady-state diffusion problem, following which Zainab implemented this
solution. Marcell created a CLI through which to run experiments, and Henry tidied up 
the plots for the report. We all contributed to the report.


= Git Fame distribution of the repository
Date and time: #json("build_time.json").datetime

Output from Git Fame:

#let git_fame_summary = csv("git_fame_summary.csv")
#let git_fame_details = csv("git_fame_detailed.csv", )

#show table.cell.where(y: 0): strong
//#figure(
#table(
		columns: 4,
		..git_fame_summary.flatten(),
	)
//)

//#figure(
#table(
		columns: 7,
		..git_fame_details.flatten(),
	)
//)
