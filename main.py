from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")
    infromation = """Abdel Fattah el-Sisi (born November 19, 1954, Cairo) is Egypt's sixth president and a former field marshal who rose through the army—serving as director of military intelligence and, in August 2012, as defense minister—before leading the removal of President Mohamed Morsi amid mass protests on July 3, 2013; he was first elected president in June 2014, won re-election in 2018, and secured a third term in December 2023 with 89.6% of the vote (official turnout 66.8%), beginning a new six-year term in April 2024 following 2019 constitutional amendments that extended presidential terms and enabled him to run again, potentially through 2030. 
 As president he has promoted large "national projects," notably the 2015 New Suez Canal expansion and a vast new administrative capital east of Cairo, while pursuing subsidy cuts, new taxes/VAT, and waves of currency devaluations tied to IMF programs; in March 2024 Egypt and the IMF expanded their arrangement to $8 billion alongside commitments to a more flexible exchange rate and slower state megaproject spending. 
 To ease a severe foreign-currency crunch, in February–April 2024 his government struck a landmark $35 billion deal with the UAE's ADQ to develop Ras El-Hekma on the Mediterranean—one of the biggest single foreign-investment packages in Egypt's history—and designated it a special free zone. 
 Sisi's decade in power has also drawn persistent criticism from rights groups and some governments for repression of dissent, mass arrests, and constrained political competition—concerns highlighted again in 2025 reporting—although high-profile pardons have occasionally occurred, such as the September 2025 release of British-Egyptian activist Alaa Abd el-Fattah. 
 Supporters credit him with restoring stability after the turmoil of 2011–2013 and with pursuing infrastructure-led growth and security cooperation; critics counter that the military-led economic model, heavy debt, and tighter civic space have magnified economic hardship and limited pluralism."""

    information_prompt = PromptTemplate(
        input_variables=["information"],
        template="""
        Give me the following information about the following person:
        {information}
        1. What is the person's name?
        """
    )

    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = information_prompt | llm
    response = chain.invoke({"information": infromation})
    print(response.content)

if __name__ == "__main__":
    main()
