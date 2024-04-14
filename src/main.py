import os
from textwrap import dedent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils import print_agent_output

from crewai import Crew, Process, Task, Agent
from llms import GPT4Turbo, ClaudeHaiku, ClaudeOpus
from datetime import datetime
from random import randint
from langchain.tools import tool

load_dotenv()

# this is a dictionary
WRITERS_TO_EMULATE = "Borges, Lovecraft, Hemingway"
TONE = "Mysterious, engaging, suspenseful"
CTA = "What if all you've been told is a lie? Follow to find the truth."
SCRIPT_DURATION_IN_SECONDS = 240
SCRIPT_DURATION_IN_WORDS = 600
REQUIREMENTS = dedent(f"""\
    **Requirements**:
        - The script should be specifically structured for a YouTube video, consisting of a voiceover narration that accompanies a series of visual scenes.

        - NO ONE TALKS EXCEPT THE NARRATOR

        - The script should have a target duration of {SCRIPT_DURATION_IN_SECONDS} seconds.

        - The script should be divided into clear, distinct scenes that
        can be easily visualized. Each scene should be described in a 
        way that translates well to a series of images or video clips.

        - Use descriptive language to help the
        viewer imagine what they'll be seeing on screen.

        - Ensure that the overall narrative flows logically from one 
        scene to the next, creating a cohesive story arc that can be
        conveyed through the combination of voiceover and visuals.

        - Consider pacing and timing. The script should be structured 
        in a way that allows for natural pauses, transitions, and 
        emphasis on key points or images.

        - Include relevant visual cues in brackets or 
        parentheses to guide the video creation process. Indicate 
        what the viewer should be seeing at each point in the script.

        Key elements to consider:
            - Tone: {TONE}
            - Writers to Emulate: {WRITERS_TO_EMULATE}
            - Call-to-Action: {CTA}
            - Script Duration: {SCRIPT_DURATION_IN_SECONDS} seconds
""")
SCRIPT_CREATION_STEPS = dedent(f"""\
    The order of the script creation process is as follows:
    0: BRIEF: big_boss
    1: RESEARCH: researcher
    2: OUTLINE: writer. PASS THE FULL RESEARCH AND FULL BRIEF TO THIS TASK!
    3: DRAFT: writer
    4: CRITIQUE: critic_editor. PASS THE FULL RESEARCH, FULL BRIEF, AND DRAFT TO THIS TASK!
    5: SCRIPT: writer
    6: SAVE: archiver. PASS THE FINAL SCRIPT TO THIS AGENT!

    Tasks should be done ONCE and NO MORE, in the order specified above.
""")
YOUTUBE_SCRIPT_REQUIREMENTS = dedent(f"""\
    1. Start with a strong hook that immediately grabs the viewer's attention and sets the tone for the story.
    2. Use concise, engaging language that is easy to follow and keeps viewers interested throughout the video.
    3. Structure the script in short, distinct sections that can be easily visualized and translated into a series of video clips or images.
    4. Incorporate cliffhangers, plot twists, or thought-provoking questions to maintain viewer engagement and encourage them to keep watching.
    5. Use descriptive language and vivid imagery to help viewers imagine the scenes and characters, but avoid overly lengthy descriptions that may not translate well to video.
    6. Include clear cues or directions for visual elements, such as [Cut to close-up of character's face] or [Montage of city scenes], to guide the video creation process.
    7. Keep the script concise and targeted to the ideal YouTube video length, {SCRIPT_DURATION_IN_SECONDS} seconds.
    8. IMAGERY SHOULD BE STATIC, NO DESCRIPTION OF ANIMATION
    9. NO TEXT ON SCREEN
    10. NO DIALOGUE. ONLY NARRATION
""")


class ScriptTasks():
    def imagine(self, agent, concept):
        return Task(
            description=dedent(f"""\
                We are creating a video about {concept}.

                This are the requirements: {REQUIREMENTS}
                
                - Define project scope, objectives, and deliverables for the script development process

                - Clearly state the goals, target audience, key messages, duration and intended use case for the script. 
                - Identify the core concept or idea that will serve as the foundation for the script, and establish the creative direction and tone for the project.
                - Consider unique angles, narrative structures, and storytelling techniques that will capture the audience's attention and effectively convey the intended message.
                - Provide the tone, writing style and a brief summary of how it expands on the original idea in an innovative and engaging way.
                - State any additional requirements or constraints
                
                - !!!: Establish an exact order of the tasks and workers to perform the script creation process. THIS IS CRITICAL
                    {SCRIPT_CREATION_STEPS}
            """),
            expected_output=dedent(f"""\
                A detailed project brief that includes:
                - The exact order of the tasks to perform
                - Core concept or idea for the script
                - Core themes and narrative elements
                - Project goals and objectives
                - Script duration
                - Target audience and key messages
                - Intended use case for the script
                - Creative direction and tone for the project
                - Any additional requirements or constraints
            """),
            agent=agent,
            async_execution=False,
        )

    def research(self, agent):
        return Task(
            description=dedent(f"""\
                Conduct in-depth research on the themes for the script, diving into the relevant subject matter, and contextual details. Gather information from your enormous wealth of knowledge with real-world examples to ensure accuracy and authenticity. Organize the research findings into a structured document.
            """),
            expected_output=dedent(f"""\
                A succint but comprehensive research document divided into sections based on key topics and themes. Each section should contain detailed information, statistics, quotes, and examples that provide a solid foundation for the script. Sources should be properly cited.
            """),
            agent=agent,
            async_execution=False,
        )
    
    def outline(self, agent):
        return Task(
            description=dedent(f"""\
                Create a detailed outline for the script based on the *brief*, the *research findings* and the given *script direction* for the project.
                
                Break down the narrative into distinct scenes or sections, describing the key events, character developments, and emotional beats. Ensure the outline has a clear beginning, middle, and end, with a logical flow and progression of ideas.

                The outline should be optimized for a video of {SCRIPT_DURATION_IN_SECONDS} seconds.
            """),
            expected_output=dedent(f"""\
                A comprehensive script outline with a hierarchical structure.
                The top level should list the major scenes or sections, with nested bullet points providing more granular details about the content and purpose of each part. The outline should read like a condensed version of the full script.
            """),
            agent=agent,
            async_execution=False,
        )
    
    def draft(self, agent):
        # Take inspiration from the following writers: {WRITERS_TO_EMULATE}
        return Task(
            description=dedent(f"""\
                Write a compelling YouTube script based on the outline.

                Remember:

                {YOUTUBE_SCRIPT_REQUIREMENTS}

                {REQUIREMENTS}

            """),
            expected_output=dedent(f"""\
                A completed first draft of the script in standard YouTube video format:
                    - Narrative voiceover
                    - Visual scene descriptions
                The draft should cover the full story arc from beginning to end, divided into scenes.
            """),
            agent=agent,
            async_execution=False,
        )

    def critique(self, agent):
        return Task(
            description=dedent(f"""\
                Provide a thorough and constructive critique of the *final draft* for the 
                script from the perspective of a professional critic. Focus
                on identifying areas for improvement and offering specific, 
                actionable suggestions to help the writer refine their work.
                
                1. Analyze the structure and pacing of the script. Does it 
                have a clear beginning, middle, and end? Is the narrative
                arc compelling and well-developed? Does the pacing keep 
                the viewer engaged or are there points where it lags?
                
                2. Evaluate the characterization and dialogue. Are the 
                characters believable and fully realized? Do they have 
                distinct voices and motivations? Does the dialogue sound
                natural and authentic or is it stilted and expository?
                
                3. Assess the themes and messaging. Is there a clear central
                theme or message? Is it effectively explored and conveyed
                throughout the script? Does it resonate on an emotional or
                intellectual level?
                
                4. Consider the originality and creativity of the concept. Is
                the idea fresh and innovative or does it feel derivative?
                Does the script bring a unique perspective or voice to the
                subject matter?
                
                5. Examine the visual storytelling and atmosphere. Does the 
                script create a strong sense of tone, mood, and ambiance?
                Are the visuals richly described and evocative? Is there
                a cohesive aesthetic vision?
                
                6. Identify any logical inconsistencies, plot holes, or 
                unanswered questions. Does the story hold together under
                scrutiny? Are there any dangling threads or unresolved
                issues that need to be addressed?

                7. Make sure research is properly incorporated.
                
                8. Make sure the original intention of the brief is met.
                
                For each point of critique, provide specific examples from 
                the script to illustrate your observations. Offer concrete
                suggestions for how the writer could address these issues and
                elevate the overall quality of the draft. Maintain a 
                constructive tone focused on improvement rather than simply
                pointing out flaws.
            """),
            expected_output=dedent(f"""
                A detailed critique of the draft script that covers the 
                following key areas:

                1. Structure and Pacing: 
                - Analysis of the narrative arc and plot development
                - Evaluation of pacing and viewer engagement
                - Specific suggestions for improving story structure

                2. Characterization and Dialogue:
                - Assessment of character development and believability
                - Review of dialogue authenticity and distinctive voices
                - Recommendations for refining characters and conversations

                3. Themes and Messaging:
                - Identification of central themes and messages
                - Evaluation of how effectively themes are conveyed
                - Suggestions for clarifying or deepening thematic resonance

                4. Originality and Creativity:
                - Assessment of the uniqueness and freshness of the concept
                - Identification of any derivative or clichéd elements
                - Suggestions for pushing the boundaries of the idea further

                5. Visual Storytelling and Atmosphere:
                - Evaluation of the script's descriptive language and imagery
                - Analysis of tone, mood, and overall aesthetic cohesion
                - Recommendations for enhancing the visual richness and impact

                6. Logical Consistency and Completeness:
                - Identification of any plot holes, inconsistencies, or gaps
                - Scrutiny of the story's internal logic and coherence
                - Suggestions for resolving unanswered questions or loose ends

                The critique should cite specific examples from the script to 
                support each point, and offer actionable suggestions for 
                improvement. The overall tone should be constructive and geared
                towards helping the writer elevate the quality of their work.
            """),
            agent=agent,
            async_execution=False,
        )
   
    def script(self, agent):
        return Task(
            description=dedent(f"""\
                Create the final, polished script, using the *finalDraft* as a foundation and the *scriptCritique* as a guide for potential improvements.

                1. Review the feedback provided in the *scriptCritique*, but remember that you have the creative freedom to decide which suggestions to incorporate. Focus on changes that resonate with your artistic vision and enhance the overall narrative, characterization, theme, and visual storytelling.

                2. Ensure that the final script is technically sound, paying attention to formatting, spelling, grammar, and adherence to the given requirements.

                3. If needed, make adjustments to the script to ensure it meets the specified duration and word count limits, but prioritize maintaining the integrity of the story and its emotional impact.

                4. If you make significant changes based on the critique, consider preparing a brief summary of those changes and the rationale behind them. However, if you choose to stick closely to your original vision, feel free to provide a short explanation of why you decided to do so.

                5. Compile the finalized script along with any relevant supplementary materials (e.g., author notes, character profiles, research references) that you feel will enhance the understanding and context of the script.

                Remember, the *scriptCritique* is just one perspective. Trust your instincts and let your unique voice shine through in the final script.

                KEEP IN MIND THE ORIGINAL REQUIREMENTS
            """),
            expected_output=dedent(f"""\
                The completed script, refined based on your creative judgment and adhering to all technical requirements.
            """),
            agent=agent,
            async_execution=False,
        )

    def saveOutput(self, agent):
        return Task(
            description="""Take the final script and save it to a markdown file.
            Your final answer MUST be a response must be showing that the file was saved .""",
            expected_output='A saved file name',
            agent=agent,
        )

class ScriptAgents():
    def big_boss(self):
        return Agent(
            role="The director and concept developer",
            goal="provide precise project briefs that guide the production of high-quality YouTube content",
            backstory=dedent(f"""
                <persona>
                    You are the Big Boss, a legendary filmmaker known for meticulous craftsmanship, 
                    deep narrative insight, and a revolutionary approach to visual storytelling. With an extensive knowledge of both classical film techniques 
                    and modern digital platforms like YouTube, you excel in structuring compelling project 
                    briefs that align with strategic vision and audience engagement.

                    YOU DON'T WRITE SCRIPTS, YOU WRITE PROJECT BRIEFS.

                    Not only that but you are a visionary leader who inspires and guides your team to produce incredible works. With a genius streak for innovation and a detailed eye for quality, you guide your coworkers, providing them with mentorship and direction. Your leadership ensures that each project aligns with high standards and strategic goals, facilitating a collaborative environment where creative ideas can flourish under your expert guidance.
                </persona>
                
                <your_work>
                    You set clear guidelines and maintain high standards for what a YouTube 
                    script should entail. This includes a strong narrative voiceover, a series of well-defined 
                    scenes, and visual storytelling that captivates and retains viewer attention. Your expertise 
                    ensures that each project brief not only defines the scope and objectives of the video content 
                    but also embeds deep artistic and thematic elements, making each piece a work of art that 
                    resonates with viewers.
                </your_work>
            """),
            llm=ClaudeHaiku,
            ##llm=GPT4Turbo,
            #memory=True
            max_iterations=1,
            #tools=human_tools,
            # step_callback=print_agent_output
            step_callback=lambda x: print_agent_output(x,"Big Boss Agent"),
            allow_delegation=False
        )
        
    def researcher(self):
        # Modeled after a fact-checker
        return Agent(
            role="Master Researcher",
            goal="To back creative concepts with thorough and relevant data",
            backstory=dedent(f"""
            You are an expert Researcher and Fact-checker. You specialize in gathering and synthesizing information to support and enrich creative concepts.
            With a background in academic research and a past role as a fact-checker for a renowned news outlet, you excel at delving into diverse topics to unearth essential truths and insights.
            
            Your mission is to provide a solid factual foundation for ideas, ensuring that narratives are authentic and grounded in reality.
            """),
            llm=ClaudeHaiku,
            max_iterations=1,
            allow_delegation=False,
            step_callback=lambda x: print_agent_output(x,"Researcher Agent")
        )
        
    def senior_writer(self):
        # Modeled after Ernest Hemingway
        return Agent(
            role="The Master Writer of narrative, focusing on thematic depth and succinct storytelling",
            goal="To write scripts",
            backstory=dedent(f"""
                You are the Master Writer: a seasoned wordsmith who crafts compelling scripts for YouTube videos. As a novelist turned YouTube scriptwriter, you have adapted your style to suit the platform, focusing on engaging storytelling that captures viewers' attention from the very first sentence.

                Your work is characterized by concise, punchy sentences that quickly get to the heart of the matter, as well as vivid descriptions and dialogue that immerse viewers in the story. You have a keen understanding of pacing and structure, crafting scripts that maintain a sense of momentum and keep viewers hooked until the very end.

                With a penchant for robust, impactful narratives, you write stories that resonate deeply with YouTube audiences, delivering your message in a way that is both direct and emotionally engaging.

                You emulate the styles of literary greats like {WRITERS_TO_EMULATE}, infusing your scripts with their unique voices and narrative techniques. Your scripts are a blend of mystery, suspense, and emotional depth, leaving viewers both entertained and intellectually stimulated.
            """),
            #llm=GPT4Turbo,
            llm=ClaudeOpus,
            memory=True,
            allow_delegation=False,
            step_callback=lambda x: print_agent_output(x,"Senior Writer Agent")
        )

    def critic_editor(self):
        return Agent(
            role="Ruthless analyst, unforgiving critic, and meticulous editor, ensuring scripts meet the highest standards of precision, depth, and artistic quality",
            goal="To relentlessly critique and edit scripts, pushing writers to elevate their work to exceptional levels",
            backstory=dedent(f"""
                You are a formidable combination of a ruthless analyst, an unforgiving critic, and a meticulous editor. With a background steeped in both the theory and practice of filmmaking, you view scripts through a lens that demands nothing less than excellence in traditional narrative structures, innovative cinematic techniques, and the power of storytelling.

                As a critic and advisor, your critiques are razor-sharp and unapologetic. You have no patience for mediocrity and are quick to point out flaws in storytelling, character development, and emotional resonance. You challenge writers to push themselves beyond their comfort zones, using your deep understanding of storytelling's nuances to guide projects towards greatness.

                As an editor, you are known as a modern-day Maxwell Perkins, but with a reputation for being uncompromising in your pursuit of perfection. You have a storied history of transforming ambitious drafts into definitive works, but only after putting writers through a gauntlet of revisions and critiques. Your keen editorial eye is both feared and respected, as you refuse to settle for anything less than the pinnacle of literary excellence.

                Your mission is to uphold your legendary status by combining your critical insights and editorial prowess to elevate every project to its maximum potential, even if it means bruising a few egos along the way. You are the gatekeeper of quality, and you will not rest until every script meets your exacting standards.
            """),
            llm=GPT4Turbo,
            max_iterations=1,
            step_callback=lambda x: print_agent_output(x, "Critic Editor Agent"),
        )

    def archiver(self):
        return Agent(
            role='saves the script to a markdown file',
            goal='Take in the final script and write it to a Markdown file',
            backstory="""You are a efficient and simple agent that gets a final script and saves it to a markdown file. in a quick and efficient manner""",
            llm=GPT4Turbo,
            verbose=True,
            step_callback=lambda x: print_agent_output(x,"Archiver Agent"),
            tools=[save_content],
        )   

@tool("save_content")
def save_content(task_output):
    """Useful to save content to a markdown file"""
    print('in the save markdown tool')
    # Get today's date in the format YYYY-MM-DD
    today_date = datetime.now().strftime('%Y-%m-%d')
    # Set the filename with today's date
    filename = f"{today_date}_{randint(0,100)}.md"
    # Write the task output to the markdown file
    with open(filename, 'w') as file:
        file.write(task_output)
        # file.write(task_output.result)

    print(f"Blog post saved as {filename}")

    return f"Blog post saved as {filename}, please tell the user we are finished"

def main():
    tasks = ScriptTasks()
    agents = ScriptAgents()

    print("# Welcome to the Saga Creative Offices")
    print("---------------------------------")

    concept = input("What is the concept you would like to develop today?")

    # Agents
    big_boss = agents.big_boss()
    researcher = agents.researcher()
    senior_writer = agents.senior_writer()
    critic_editor = agents.critic_editor()
    archiver = agents.archiver()

    # Tasks
    brief = tasks.imagine(big_boss, concept)
    researchFindings = tasks.research(researcher)
    outline = tasks.outline(senior_writer)
    draft = tasks.draft(senior_writer)
    scriptCritique = tasks.critique(critic_editor)
    script = tasks.script(senior_writer)
    saveOutput = tasks.saveOutput(archiver)

    researchFindings.context = [brief]
    outline.context = [brief, researchFindings]
    draft.context = [brief, outline]
    scriptCritique.context = [draft, brief, researchFindings]
    script.context = [draft, scriptCritique,]
    saveOutput.context = [script]


    # Crew
    crew = Crew(
        agents=[
            big_boss,
            researcher,
            senior_writer,
            critic_editor,
            archiver,
        ],
        tasks=[
            brief,
            researchFindings,
            outline,
            draft,
            scriptCritique,
            script,
            saveOutput,
        ],
        #manager_llm=ClaudeOpus,
        manager_llm=GPT4Turbo,
        process=Process.sequential,
        memory=False,
        verbose=2,
        step_callback=lambda x: print_agent_output(x,"MasterCrew Agent")
    )

    result = crew.kickoff()

    print(result)

if __name__ == "__main__":
    main()