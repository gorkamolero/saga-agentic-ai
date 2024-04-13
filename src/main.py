from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from crewai import Crew
from tasks import ScriptTasks
from agents import ScriptAgents

def main():
    load_dotenv()
    
    tasks = ScriptTasks()
    agents = ScriptAgents()

    print("# Welcome to the Saga AI offices")
    print("---------------------------------")

    concept = input("What is the concept you would like to develop today? Give us a brief overview of your idea: ")

    # Agents
    concept_developer = agents.concept_developer()
    researcher = agents.researcher()
    staff_writer = agents.staff_writer()
    gen_z_viralizer = agents.gen_z_viralizer()
    senior_writer = agents.senior_writer()
    critic = agents.critic()
    senior_editor = agents.senior_editor()

    # Tasks
    scriptDirection = tasks.imagine(concept_developer, concept)
    researchFindings = tasks.research(researcher, concept)
    outline_task = tasks.outline(staff_writer, concept)
    firstDraft = tasks.firstDraft(staff_writer, concept)
    factCheck = tasks.factCheck(researcher)
    viralDraft = tasks.viralize(gen_z_viralizer, concept)
    finalDraft = tasks.finalDraft(senior_writer)
    scriptCritique = tasks.critique(critic)
    script = tasks.script(senior_editor)

    outline_task.context = [researchFindings, scriptDirection]
    firstDraft.context = [outline_task]
    factCheck.context = [firstDraft, researchFindings]
    viralDraft.context = [firstDraft]
    finalDraft.context = [viralDraft, factCheck]
    scriptCritique.context = [finalDraft]
    script.context = [finalDraft, scriptCritique]

    # Crew
    crew = Crew(
        agents=[
            concept_developer,
            researcher,
            staff_writer,
            gen_z_viralizer,
            senior_writer,
            critic,
            senior_editor
        ],
        tasks=[
            scriptDirection,
            researchFindings,
            outline_task,
            firstDraft,
            factCheck,
            viralDraft,
            finalDraft,
            scriptCritique,
            script
        ],
        manager_llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        process="hierarchical",
        memory=True
    )

    result = crew.kickoff()

    print(result)

if __name__ == "__main__":
    main()