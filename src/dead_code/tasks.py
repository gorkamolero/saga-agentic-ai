
    
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