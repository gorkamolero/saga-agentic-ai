import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from tasks import ScriptTasks
from agents import ScriptAgents

load_dotenv()

def main():
    tasks = ScriptTasks()
    agents = ScriptAgents()

    print("# Welcome to the Saga AI offices")
    print("---------------------------------")

    concept = input("What is the concept you would like to develop today? Give us a brief overview of your idea: ")

    # Agents
    big_boss = agents.big_boss()
    researcher = agents.researcher()
    staff_writer = agents.staff_writer()
    gen_z_viralizer = agents.gen_z_viralizer()
    senior_writer = agents.senior_writer()
    critic = agents.critic()
    senior_editor = agents.senior_editor()

    # Tasks
    brief = tasks.imagine(big_boss, concept)
    researchFindings = tasks.research(researcher)
    outline = tasks.outline(staff_writer)
    firstDraft = tasks.firstDraft(staff_writer)
    factCheck = tasks.factCheck(researcher)
    viralDraft = tasks.viralize(gen_z_viralizer)
    finalDraft = tasks.finalDraft(senior_writer)
    scriptCritique = tasks.critique(critic)
    script = tasks.script(senior_editor)
    theFinalTouch = tasks.theFinalTouch(big_boss)

    researchFindings.context = [brief]
    outline.context = [brief, researchFindings]
    firstDraft.context = [brief, outline]
    factCheck.context = [firstDraft, researchFindings]
    viralDraft.context = [firstDraft]
    finalDraft.context = [viralDraft, factCheck, brief]
    scriptCritique.context = [finalDraft]
    script.context = [finalDraft, scriptCritique, brief]
    theFinalTouch.context = [script, brief]


    # Crew
    crew = Crew(
        agents=[
            big_boss,
            researcher,
            staff_writer,
            gen_z_viralizer,
            senior_writer,
            critic,
            senior_editor
        ],
        tasks=[
            brief,
            researchFindings,
            outline,
            firstDraft,
            factCheck,
            viralDraft,
            finalDraft,
            scriptCritique,
            script,
            theFinalTouch
        ],
        #manager_llm=ClaudeOpus,
        #manager_llm=ChatOpenAI(
        #    temperature=0,
        #    model="gpt-4-turbo"
        #),
        #process=Process.hierarchical,
        #memory=True
    )

    result = crew.kickoff()

    print(result)

if __name__ == "__main__":
    main()