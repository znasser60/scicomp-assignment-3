= Team effort distribution report

#show link: set text(fill: blue)

*Assignment number:* 3

*Assignment team number:* 12

*GitHub URL:* #link("https://github.com/znasser60/scicomp-assignment-3")


= Git Fame distribution of the repository
Date and time: #json("effort_distribution.json").datetime

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
