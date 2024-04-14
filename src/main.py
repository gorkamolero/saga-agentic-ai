import os
from textwrap import dedent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils import print_agent_output

from crewai import Crew, Process, Task, Agent
from llms import GPT4Turbo, ClaudeHaiku
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

        - The script should be approximately {SCRIPT_DURATION_IN_WORDS} words, with a target duration of {SCRIPT_DURATION_IN_SECONDS} seconds when read aloud.

        - When crafting the script, keep the following requirements in mind:
            - The script should be divided into clear, distinct scenes that
            can be easily visualized. Each scene should be described in a 
            way that translates well to a series of images or video clips.

            - The narration should be written in a conversational, engaging
            style that complements the visuals and keeps the viewer 
            interested throughout. Use descriptive language to help the
            viewer imagine what they'll be seeing on screen.

            - Ensure that the overall narrative flows logically from one 
            scene to the next, creating a cohesive story arc that can be
            conveyed through the combination of voiceover and visuals.

            - Consider pacing and timing. The script should be structured 
            in a way that allows for natural pauses, transitions, and 
            emphasis on key points or images.

            - Include relevant visual cues or directions in brackets or 
            parentheses to guide the video creation process. Indicate 
            what the viewer should be seeing at each point in the script.

        - The script should have clear scenes that make sense in a visual format. It should be a narrative that can be visualized.

        Key elements to consider:
            - Tone: {TONE}
            - Writers to Emulate: {WRITERS_TO_EMULATE}
            - Call-to-Action: {CTA}
            - Script Duration: {SCRIPT_DURATION_IN_SECONDS} seconds

        The order of the script creation process is as follows:
            0: BRIEF: big_boss
            1: RESEARCH: researcher
            2: OUTLINE: writer
            3: FIRST DRAFT: writer
            4: CIRCULARIZE: gen_z_viralizer
            6: FINAL DRAFT: writer
            7: CRITIQUE: critic_editor
            8: SCRIPT: writer
""")

class ScriptTasks():
    def imagine(self, agent, concept):
        return Task(
            description=dedent(f"""\
                We are creating a video about {concept}.

                This are the requirements: {REQUIREMENTS}
                
                - Define project scope, objectives, and deliverables for the script development process
                - Establish an exact order of the tasks and workers to perform the script creation process. THIS IS CRITICAL
                    1: RESEARCH: researcher
                    2: OUTLINE: writer
                    3: FIRST DRAFT: writer
                    4: CIRCULARIZE: gen_z_viralizer
                    5: HOOKIFY: gen_z_viralizer
                    6: FINAL DRAFT: writer
                    7: CRITIQUE: critic_editor
                    8: SCRIPT: writer

                - Clearly state the goals, target audience, key messages, duration and intended use case for the script. 
                - Identify the core concept or idea that will serve as the foundation for the script, and establish the creative direction and tone for the project.
                - Consider unique angles, narrative structures, and storytelling techniques that will capture the audience's attention and effectively convey the intended message.
                - Provide the tone, writing style and a brief summary of how it expands on the original idea in an innovative and engaging way.
                - State any additional requirements or constraints
                
                Provide your reasons and a step-by-step explanation on how you achieved these results.
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
                Write the first draft of the YouTube video script based on the *brief*, the *outline*, the *research findings*.
                
                Craft a compelling monologue, with rich descriptions, and emotive language to bring the story, characters and narrative to life. Focus on getting the complete narrative down without worrying too much about perfection at this stage.

                The script should have clear scenes that make sense in a visual format. It should be a narrative that can be visualized.
            
                The script should be optimized for a video of {SCRIPT_DURATION_IN_SECONDS} seconds.

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

    def circularize(self, agent):
        return Task(
            description=dedent(f"""\
                Restructure the script to have a circular narrative that
                connects the end back to the beginning in a clever and
                satisfying way. This could involve setting up a question or
                mystery at the start that gets answered or resolved at the
                end, creating a sense of closure and completeness.
            """),
            expected_output=dedent(f"""\
                A revised script with a circular narrative structure that circles back on itself, with the ending connecting to the beginning in a meaningful way
                that creates a sense of completeness and closure.
            """),
            max_iterations=2,
        )

    def hookify(self, agent):
        return Task(
            description=dedent(f"""\
                Identify the most compelling, emotionally resonant elements of the script. Remember these key moments and themes.
                
                Then, enhance the script by adding a compelling hook at the beginning
                that grabs the viewer's attention and draws them into the story.
                
                The hook should be intriguing, engaging, and relevant to the
                core themes or ideas of the script. It should set the tone for
                the rest of the video and make the viewer eager to continue
                watching.
            """),
            expected_output=dedent(f"""\
                An ttention-grabbing hook that draws the viewer in and sets the stage for the rest of the video. The hook should be compelling, relevant, and engaging, creating a sense of intrigue and anticipation.
            """),
            agent=agent,
            async_execution=False,
        )
     
    def finalDraft(self, agent):
        return Task(
            description=dedent(f"""\
                Review the given enhancements into a better script, by incorporating the *critique* feedback. Ensure the script is polished, cohesive, and engaging, ready for the final review.

                Remember the requirements: {REQUIREMENTS}

                1. First and foremost, assess the overall narrative cohesion 
                and creative tone of the script. Does the story flow 
                logically and engagingly from beginning to end? Is the 
                unique voice and style of the piece consistently maintained
                throughout? These are the most critical elements to get right.

                2. Consider the key points from the critic's review, especially
                those related to structure, characterization, theme, and 
                visual storytelling. Evaluate whether the script has 
                sufficiently addressed any major concerns in these areas,
                while still preserving the core creative vision.

                3. Remain open to the critic's suggestions for improvement, but
                don't feel obligated to incorporate every piece of feedback.
                Some notes may be subjective or not fully aligned with the
                script's intentions. Trust your instincts and make changes
                only where you feel they genuinely enhance the work.

                4. Once you're satisfied with the content, do a final technical
                review. Double check formatting, page length, spelling, and
                grammar. Make any necessary adjustments to ensure the script
                is polished and professional.

                5. Package the script with any relevant supplementary materials
                such as author notes, character breakdowns, or research
                references. Ensure these materials are clearly organized and
                enhance the reader's understanding of the script.

                The goal is to arrive at a final draft that is narratively
                cohesive, creatively powerful, and technically impeccable. The
                critic's feedback should inform but not dictate the ultimate
                shape of the work. Prioritize changes that elevate the script's
                core strengths and unique voice.
            """),
            expected_output=dedent(f"""\
                The packaged, 100% finalized script, including:

                1. The completed script file in the appropriate format for the
                use case, reflecting any final content or technical edits 
                based on a careful consideration of the critic's feedback.

                2. A short memo that includes:
                - Confirmation that the script is locked and ready for the
                    next stage of the process.
                - A brief summary of any significant changes made in 
                    response to the critic's feedback, and the rationale
                    behind those changes.
                - Acknowledgement of any feedback that was considered but
                    ultimately not incorporated, and the reasons for those
                    decisions.
                - A final assessment of how the script achieves its goals in
                    terms of narrative cohesion, creative tone, and overall 
                    impact.

                3. Any relevant supplementary materials, such as:
                - Author notes or a creator statement
                - Character breakdowns or profiles
                - Research references or source materials
                - A brief synopsis or logline

                These materials should be clearly organized and labeled in a 
                way that enhances the reader's engagement with and understanding
                of the script. The entire package should represent a polished, 
                professional, and compelling final product.
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

                Keep the requirements in mind: {REQUIREMENTS}
            """),
            expected_output=dedent(f"""\
                The final, polished script package, including:

                1. The completed script file, refined based on your creative judgment and adhering to all technical requirements.

                2. If significant changes were made based on the *scriptCritique*, a brief summary of those changes and the rationale behind them. If you chose to stick closely to your original vision, a short explanation of why.

                3. Any supplementary materials (e.g., author notes, character profiles, research references) that you feel will enhance the understanding and context of the script.

                The final package should represent a compelling, professionally executed script that showcases your unique voice and creative vision.
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
            role="Expert Researcher and Factchecker",
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
                You are the Master Writer: a seasoned wordsmith who elevates scripts to their full potential.
                As a novelist turned scriptwriter, you have always favored a style
                marked by its economy of words and its depth of emotion.
                Your work is characterized by its vivid, yet straightforward
                descriptions and dialogue that cuts to the heart of human experiences.

                Your mission is to strip back unnecessary elements from our scripts,
                focusing on strong, simple language and clear, impactful themes.
                With a penchant for robust, impactful narratives, you refine our stories
                to ensure they are potent and resonate deeply with audiences,
                while remaining refreshingly direct.
            """),
            #llm=GPT4Turbo,
            llm=ClaudeHaiku,
            memory=True,
            allow_delegation=False,
            step_callback=lambda x: print_agent_output(x,"Senior Writer Agent")
        )

    def gen_z_viralizer(self):
        # Modeled after a young social media influencer
        return Agent(
            role="Young genius, expert in viral video structures, social media algorithms and engagement psychology and theory",
            goal="To steer and refine scripts for perfect engagement with modern platforms and audiences.",
            backstory=dedent(f"""
                You are a digital native who first went viral at age 14. A young genius expert in viral video structures, social media algorithms and engagement psychology and theory.
                Your sharpness and depth of intellect and creativity has 
                allowed you to work with the best content creators in the world.

                Your mission is to modify scripts by using your knowledge
                of social media algorithms and engagement psychology and tricks
                to ensure they resonate with the target audience and maximize engagement.
            """),
            llm=ClaudeHaiku,
            max_iterations=2,
            memory=False,
            #tools=human_tools,
            step_callback=lambda x: print_agent_output(x,"Gen Z Viralizer Agent"),
            allow_delegation=False
        )

    def critic_editor(self):
        return Agent(
            role="Expert analyst, critic, advisor, and legendary editor, ensuring precision, depth, and artistic quality",
            goal="To enhance the narrative, technical, and emotional aspects of scripts through perceptive critiques and meticulous editing",
            backstory=dedent(f"""
                You are a unique blend of an expert analyst, critic, advisor, and legendary editor. With a background steeped in both the theory and practice of filmmaking, you view scripts through a lens that appreciates traditional narrative structures, innovative cinematic techniques, and the power of storytelling.

                As a critic and advisor, your critiques blend a deep appreciation of storytelling with a keen eye for directorial flair and the ability to connect emotionally with an audience. You challenge and inspire writers to elevate their work, using your understanding of storytelling's nuances to guide projects that not only entertain but linger in the minds and emotions of viewers.

                As an editor, you are known as a modern-day Maxwell Perkins, with a storied history of transforming ambitious drafts into definitive works. Your reputation as a nurturing yet incisive editor draws both budding and seasoned writers who seek your mentorship and keen editorial eye. With a meticulous eye for detail and a profound understanding of storytelling, you refine scripts to meet the pinnacle of literary excellence, ensuring they resonate deeply with audiences and secure their place as memorable and impactful works.

                Your mission is to uphold your legendary status by combining your critical insights and editorial prowess to elevate every project to its maximum potential, maintaining a legacy of quality and success.
            """),
            llm=ClaudeHaiku,
            max_iterations=1,
            #tools=human_tools
            step_callback=lambda x: print_agent_output(x,"Critic Editor Agent"),

        )

    def archiver(self):
        return Agent(
            role='Final Script Archiver',
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
    gen_z_viralizer = agents.gen_z_viralizer()
    senior_writer = agents.senior_writer()
    critic_editor = agents.critic_editor()
    archiver = agents.archiver()

    # Tasks
    brief = tasks.imagine(big_boss, concept)
    researchFindings = tasks.research(researcher)
    outline = tasks.outline(senior_writer)
    firstDraft = tasks.draft(senior_writer)
    secondDraft = tasks.circularize(gen_z_viralizer)
    hookify = tasks.hookify(gen_z_viralizer)
    finalDraft = tasks.finalDraft(senior_writer)
    scriptCritique = tasks.critique(critic_editor)
    script = tasks.script(senior_writer)
    saveOutput = tasks.saveOutput(archiver)

    researchFindings.context = [brief]
    outline.context = [brief, researchFindings]
    firstDraft.context = [brief, outline]
    secondDraft.context = [firstDraft]
    hookify.context = [secondDraft]
    finalDraft.context = [hookify, secondDraft]
    scriptCritique.context = [hookify, firstDraft]
    script.context = [finalDraft, scriptCritique, brief]
    saveOutput.context = [script]


    # Crew
    crew = Crew(
        agents=[
            big_boss,
            researcher,
            gen_z_viralizer,
            senior_writer,
            critic_editor,
            archiver,
        ],
        tasks=[
            brief,
            researchFindings,
            outline,
            firstDraft,
            secondDraft,
            hookify,
            finalDraft,
            scriptCritique,
            script,
            saveOutput,
        ],
        #manager_llm=ClaudeOpus,
        manager_llm=GPT4Turbo,
        process=Process.hierarchical,
        memory=False,
        verbose=2,
        step_callback=lambda x: print_agent_output(x,"MasterCrew Agent")
    )

    result = crew.kickoff()

    print(result)

if __name__ == "__main__":
    main()