<style>
/* =========================
   Global Styles
   ========================= */

/* Adjust the base font size here: */
body {
  font-size: 18px;  /* <--- Modify this value to increase/decrease font size */
  font-family: "Helvetica", Arial, sans-serif;
  line-height: 1.6;
  color: #333;
  margin: 1em;
}

/* Spacing around paragraphs: */
p, ul, ol {
  margin-bottom: 1em;
}

/* Make headings visually distinct: */
h1, h2, h3, h4, h5, h6 {
  color: #85c9fd;
  font-weight: bold;
  margin-top: 1.5em;
  margin-bottom: 0.5em;
}

/* Bold text with a subtle color: */
strong {
  color: #85c9fd;
}

/* Style for bullet points: */
ul {
  margin-left: 1.5em;
  list-style-type: disc;
}

/* A bit of styling for code blocks or inline code: */
pre, code {
  background-color: #f8f9fa;
  padding: 0.4em 0.6em;
  border: 1px solid #e2e3e5;
  border-radius: 4px;
  font-family: "Courier New", Courier, monospace;
}

/* Add a subtle border around images: */
img {
  border: 1px solid #ccc;
  border-radius: 4px;
  max-width: 100%;
}
</style>

# rSTAR meets ARC
## Weekly progress report

### Week 1 (03.03.2025)

- **Work planned**: 
  - Do some more research on ARC and rSTAR
  - Start coding up a bare-bones working version of "Round 1"
    - Decide how to break down code into classes and functions
    - Defined "step format" in tree

- **Work done**: 
  - Research
  - Started breaking down code into classes and functions
  - Getting a cold (again) in the first week of the project :(
  - Being plagued by needing to constantly make decisions and trade-offs

- **Issues and Questions**:
  - Probably LLM fine-tuning and alignment needed before "Round 1" is feasible
  - Would a reasoning model help with making sense of longer prompts and possibly give answers that have the correct format more often?
  - Discuss "step format" in more detail.

---

### Week 2 (10.03.2025)

- **Work planned**: 
  - Finish coding up a bare-bones working version of "Round 1"
  - Clean up code and make it as "extendable" as possible

- **Work done**: 
  - Decided on not specifying the extent of the step format beyond the "python block" limitation (that means the prefix code of every step must be valid python code)
  - Successfully implemented working version of "Round 1" with beam search agent (also successfully solved first ARC Task)
  - Rewriting code to adhere to HCP Storage best practices (i.e. use local scratch space for all intermediate files)
  - Wrote *tree visualizer* to help with debugging (and because it's cool)
  - Played firefighter (fire=bugs) basically for a week straight (I'm not complaining, I love it)
  - Wrote most of the MCTS implementation (should be functional now, but hasn't been thoroughly tested yet)
  - Wrote and rewrote the code execution environment (the second version now uses subprocesses which is more scalable and now I have more fine-grained control over resources)
  - **Finished a first rudimentary proof of concept!!!**

- **Issues and Questions**:
  - It seems the SLMs have very weak performance with the current system. What is the best way to mitigate this issue?
  - How do I effectively optimize the prompt for the policy SLM?
  - Sometimes the LLMs freeze up if I submit multiple jobs?
  - The net scratch seems to be overloaded a lot of times ...

- **Visuals**:
  - Tree visualization output for beam search with branching factor 3 and width 3  
    ![img1.png](images/week2_bs_visualization.png)

  - Tree visualization output for mcts 16 rollouts and branching factor 3  
    ![img2.png](images/week2_mcts_visualization.png)

---

### Week 3 (17.03.2025)

- **Work planned**:
  - Check for errors and bugs (mainly in MCTS implementation) and fix them
  - Experiment with different prompts and prompt formats for the policy SLM
  - Test the "best" versions on more ARC tasks and see how they perform
  - Curate a set of "very easy" tasks to effectively generate training data for the policy SLM
  - Think about how to use reasoning models in the current system
  - Think about better step definitions that allow for more comprehensive intermediate code execution
    - Also try to do more sophisticated code analysis and intermediate code execution (this should come at virtually no runtime cost)
  - **Major code refactoring for maintainability** (I will probably try to take some more inspiration from the rStar-Math implementation)
  - *Do research on fine-tuning LLMs*
  - *Write code for LLM fine-tuning*
  - *Potentially look into using a bigger model for the initial fine-tuning data generation*

- **Work done**:
  - (To be updated soon)

- **Issues and Questions**:
  - (To be updated soon)
