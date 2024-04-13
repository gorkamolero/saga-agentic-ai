from textwrap import dedent
from crewai import Task
from config import REQUIREMENTS, SCRIPT_DURATION_IN_SECONDS
#from config import WRITERS_TO_EMULATE, TONE, CTA, SCRIPT_DURATION_IN_SECONDS

class ScriptTasks():
    def imagine(self, agent, concept):
        return Task(
            description=dedent(f"""\
                We are creating a video about {concept}.
                
                First, define the project scope, objectives, and deliverables for the script development process:
                - Clearly outline the goals, target audience, key messages, duration and intended use case for the script. 
                - Identify the core concept or idea that will serve as the foundation for the script, and establish the creative direction and tone for the project.
                - Pre-requirements: {REQUIREMENTS}
                
                Then, generate a creative and engaging way to develop this concept into a compelling script for a YouTube video of approximately {SCRIPT_DURATION_IN_SECONDS} seconds:
                - Consider unique angles, narrative structures, and storytelling techniques that will capture the audience's attention and effectively convey the intended message.
                - Build upon the original idea by exploring different possibilities for character arcs, plot twists, visual motifs, and thematic depth. Aim to create a fresh and memorable take on the core concept that will resonate with an online audience.
                - Provide a high-level script direction, with tone and writing style, and a brief summary of how it expands on the original idea in an innovative and engaging way.
                
                Provide your reasons and a step-by-step explanation on how you achieved these results.
            """),
            expected_output=dedent("""\
                A detailed project brief and script direction that includes:
                - Core concept or idea for the script
                - Core themes and narrative elements
                - Project goals and objectives
                - Target audience and key messages
                - Intended use case for the script
                - Creative direction and tone for the project
                - Any additional requirements or constraints
                - An detailed description of an imagined script, described in 2-3 sentences. It should take the original idea in a distinct and compelling direction, demonstrating creative approaches to character, plot, theme, and style that will engage the target audience. The direction should feel fresh and true to the spirit of the original concept. Remember the duration: {SCRIPT_DURATION_IN_SECONDS} seconds.
            """),
            agent=agent,
            async_execution=False,
        )


    def research(self, agent):
        # add: Tone: {TONE}
        return Task(
            description=dedent(f"""\
                Conduct in-depth research on the themes *brief*, diving into the relevant subject matter, themes, and contextual details. Gather information from your enormous wealth of knowledge with real-world examples to ensure accuracy and authenticity. Organize the research findings into a structured document.
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
    
    def firstDraft(self, agent):
        # Take inspiration from the following writers: {WRITERS_TO_EMULATE}
        return Task(
            description=dedent(f"""\
                Write the first draft of the script based on the *brief*, the *outline*, the *research findings*, and the given *script direction*.
                
                Craft a compelling monologue to be read by a single voice actor, with rich descriptions, and emotive language to bring the story, characters and narrative to life. Focus on getting the complete narrative down without worrying too much about perfection at this stage.

                The script should have clear scenes that make sense in a visual format. It should be a narrative that can be visualized.
            
                The script should be optimized for a video of {SCRIPT_DURATION_IN_SECONDS} seconds.

            """),
            expected_output=dedent(f"""\
                A completed first draft of the script in standard screenplay or narrative format, depending on the project type. The draft should cover the full story arc from beginning to end, divided into scenes or chapters as appropriate. Words, scenes, and descriptions should be included throughout.
            """),
            agent=agent,
            async_execution=False,
            human_input=True,
        )
    
    def factCheck(self, agent):
        return Task(
            description=dedent(f"""\
                Carefully review the *first draft* of the script and the *research findings* to ensure all factual
                information is accurate and properly sourced. Double check
                names, dates, locations, scientific or historical details, and
                any other factual claims against the research and additional
                authoritative sources as needed. Make note of any inaccuracies
                and suggest corrections.
            """),
            expected_output=dedent(f"""\
                A fact-check report listing any inaccuracies found in the 
                draft script, along with the correct information and sources.
                The report should clearly reference the relevant page/line 
                numbers in the script. If no inaccuracies are found, the report
                should indicate that the script passed the fact-check.
            """),
            agent=agent,
            async_execution=False,
        )
    
    def viralize(self, agent):
        #Call-to-Action: {CTA}
        return Task(
            description=dedent(f"""\
                Enhance the *first draft* of the script to make it more relatable and engaging 
                for modern audiences:

                1. Identify the most compelling, emotionally resonant elements 
                of the script that will capture viewer attention and 
                interest. Highlight these key moments and themes.

                3. Restructure the script to have a circular narrative that
                connects the end back to the beginning in a clever and
                satisfying way. This could involve setting up a question or
                mystery at the start that gets answered or resolved at the
                end, creating a sense of closure and completeness. Or, you
                could use bookending scenes, recurring motifs, or callbacks
                to earlier moments to tie the whole story together.

                3. Craft a compelling, attention-grabbing hook that will draw viewers in and make them want to watch more. This could be a surprising fact, a thought-provoking question, a bold  statement, or an intriguing teaser of what's to come. The hook should be short, punchy, and memorable.

                4. Adjust the language to be more conversational and natural, 
                as if speaking directly to the viewer. Use contractions, 
                simplify complex phrasing, and aim for a warm, relatable 
                tone. The script should sound like it's coming from a 
                friendly, knowledgeable narrator.

                5. Where appropriate, try to evoke strong visuals and tap into
                the viewer's imagination. Use descriptive language and 
                metaphors to paint a picture in the audience's mind and help
                them feel more immersed in the story.

                6. Add a call to action where appropriate, according to the tone of the script.
            """),
            expected_output=dedent(f"""\
                A viral-optimized version of the script with the following:

                1. A short, punchy, irresistible hook at the beginning that 
                immediately grabs the viewer's attention and entices them
                to continue watching by promising something intriguing.

                2. A new narrative structure that circles back on itself, with
                the ending connecting to the beginning in a meaningful way
                that creates a sense of completeness and closure. Clearly
                show how the script has been restructured to achieve this
                circular effect.

                3. Key moments and themes likely to resonate with modern viewers
                clearly identified and woven throughout the script. These
                should be elements that will evoke emotion, pique curiosity, 
                or get people talking.

                4. Strategically placed references to relevant cultural trends
                or experiences that will help today's audiences connect with
                the content. These should not detract from the core story.

                5. Language that is accessible, engaging, and sounds natural, 
                like a friendly expert having a direct conversation with the
                viewer. The tone should be warm and down-to-earth.

                6. Vivid visuals evoked through expressive, imaginative language
                that transports the viewer and makes the story come alive in
                their mind. Show descriptive details and metaphors used to
                enhance the story's impact.

                The revised script should open with a strong hook, take the
                viewer on a cohesive journey that circles back to the start,
                resonate with the target audience, paint an immersive picture,
                and maintain the integrity of the original idea.
            """),
            agent=agent,
            async_execution=False,
        )
    
    def finalDraft(self, agent):
        return Task(
            description=dedent(f"""\
                Revise the given *viral draft* into a better script, by incorporating the
                *fact-check* corrections as appropriate and circling back to the *brief*. Tighten up the pacing, clarify any confusing
                points, and look for opportunities to heighten the emotional
                impact. Ensure the script feels polished and ready for final
                review.
            """),
            expected_output=dedent(f"""\
                A completed second draft of the script, fully revised and
                refined based on the previous rounds of review and ideation.
                The draft should be properly formatted, free of errors and
                typos, and ready for executive review and feedback.
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
                Review the *final draft* of the script to ensure it is ready for the intended 
                use case (production, publishing, etc.), while keeping in mind 
                the *critique* / feedback.

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
            human_input=True,
        )
    
    def theFinalTouch(self, agent):
        return Task(
            description=dedent(f"""\
                Circling back to the first *brief*, provide the final touch to the *final script*, ensuring it is fully ready. Review the script for any last-minute adjustments, polish, or enhancements that will elevate the overall quality and impact of the work.

                1. Perform a final read-through of the script to catch any lingering errors, inconsistencies, or areas for improvement. Pay close attention to spelling, grammar, punctuation, and formatting to ensure the script is error-free and professional.

                2. Consider the script as a whole and evaluate its narrative coherence, emotional resonance, and creative impact. Look for opportunities to enhance key moments, deepen characterizations, or refine dialogue to make the script more engaging and memorable.

                3. Address any remaining feedback or suggestions from previous rounds of review, ensuring that all relevant points have been incorporated or addressed to the best of your ability. Make final adjustments based on this feedback to ensure the script meets the highest standards of quality.

                4. Check the script against the original project brief and objectives to confirm that it aligns with the initial vision and goals for the work. Ensure that the script fulfills its intended purpose and effectively communicates the core concept or idea to the target audience.

                5. Prepare the script for delivery by packaging it with any necessary supplementary materials, such as author notes, character breakdowns, or research references. Organize these materials in a clear and professional manner to enhance the reader's understanding and appreciation of the script.

                The final touch should elevate the script to its highest potential, ensuring that it is polished, professional, and ready for its intended use. By carefully reviewing and refining the script, you can create a compelling and impactful work that resonates with the audience and achieves its creative goals.
            """),
            expected_output=dedent(f"""\
                A fully polished and refined version of the script that is ready for production, publishing, or any other intended use case. The script should be error-free, professionally formatted, and engaging to the target audience. It should reflect the highest standards of quality and creativity, meeting or exceeding the initial project brief and objectives.

                The final script package should include:
                - The completed script file in the appropriate format for the use case, with all necessary edits and adjustments made.
                - A short memo summarizing the final touch process, including any significant changes or enhancements made to the script.
            """),
            agent=agent,
            async_execution=False,
        )