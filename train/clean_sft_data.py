# TODO: implement a function to clean the sft data


# open raw.jsonl file

# read a batch of lines and parse it as json

# for each task in batch, get the corresponding task and load the task file

# using processpool, for each solution, try leaving out steps and see if the solution is still valid
# if it is, remove the step from the solution
# if not, keep the step in the solution
# do this in a greedy way, so that we can remove as many steps as possible
# save the cleaned solution to a new file: cleaned.jsonl
