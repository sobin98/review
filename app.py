# app.py

import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import random


# 감성 키워드 강조 함수
def highlight_keywords(text, keywords):
    for kw in keywords:
        text = text.replace(kw, f"**{kw}**")  # bold 처리
    return text

# 장르별 감성 키워드
genre_keywords = {
    "thriller":["Kill","thriller","love","play","killer","perfect","original","well","bad","pretty","amazing","violence","interesting","truly","sure","DieHard","masterpiece","excellent","want","top"],
    "drama":["love","play","war","beautiful","true","well","perfect","emotional","masterpiece","interesting","want","brilliant","heart","truly","excellent","kind","friend","amazing","wonderful","hard"],
    "horror":["horror","special","original","perfect","love","greatest","favorite","truly","tension","well","true","amazing","masterpiece","paranoia","effective","create","scare","trust","fear","terror"],
    "action":["War","love","adventure","original","amazing","bad","perfect","special","fun","villain","well","battle","want","Marvel","loved","excellent","fight","good","pretty","truly"],
    "comedy":["comedy","love","funny","fun","humor","laugh","perfect","well","joke","hilarious","play","friend","entertaining","want","wonderful","bad","humour","sure","top","beautiful"],
    "romance":["Beauty","love","beautiful","true","heart","perfect","romance","lost","favorite","brilliant","worth","enchanted","villain","hope","heroine","wonderful","romantic","amazing","care","masterpiece"],
    "adventure":["Spirited","original","beautiful","love","amazing","interesting","war","want","free","wonderful","witch","friend","strange","favorite","masterpiece","kind","hard","truly","greatest","stunning"],
    "sci_fi":["emotional","beautiful","feeling","heart","original","love","recommend","hope","heartwarming","help","adventure","truly","play","humor","create","loved","stunning","masterpiece","leave","true"],
    "biography":["love","interesting","true","friend","important","truly","hard","hope","greatest","emotional","struggle","well","want","hero","truth","inspiring","fan","accident","beautiful","war"],
    "crime":["murder","killing","death","truth","important","crime","guilty","crisis","powerful","killer","fascinating","truly","hard","problem","matter","support","credit","free","justice","murdered"],
    "animation":["true","friend","kind","friendship","love","beautiful","emotional","tears","well","respect","played","illness","affected","playing","cry","share","hard","beautifully","heartwarming","heartbreaking"],
    "family":["dream","beautiful","loved","problem","amazing","Love","Well","want","beautifully","interested","truly","help","Festival","missing","interesting","important","hard","true","natural","dead"]

}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_generator():
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer

model, tokenizer = load_generator()

# 앱 UI 시작
st.title("🎬 AI 영화 리뷰 생성기")

title = st.text_input("🎞 영화 제목", "Parallel Dream")
rating = st.slider("⭐ 평점 (1~10)", 1.0, 10.0, 8.8)

# 감독 목록
director_list = ['Christopher McQuarrie', 'Dean Fleischer Camp', 'Zach Lipovsky', 'Guy Ritchie', 'Ryan Coogler', 'Matt Palmer', 'Jake Schreier', 'Stephen Chbosky', 'Bong Joon Ho', 'David F. Sandberg', 'Jared Hess', 'James Gunn', 'David Ayer', 'Marc Webb', 'James Wong', 'Emilie Blichfeldt', 'Wes Anderson', 'Dean DeBlois', 'Alex Garland', 'Andrew DeYoung', 'Brian De Palma', 'Gareth Edwards', 'Gareth Evans', "Gavin O'Connor", 'Lynne Ramsay', 'Jonathan Entwistle', 'Trey Edward Shults', 'Brady Corbet', 'Len Wiseman', 'Isaiah Saxon', 'Steven Soderbergh', 'Drew Hancock', 'Dan Berk', 'Edward Berger', 'Ridley Scott', 'Danny Boyle', 'J.J. Abrams', 'David R. Ellis', 'Chris Sanders', 'Brad Bird', 'Ari Aster', 'Jon Avnet', 'John Woo', 'Danny Philippou', 'James Mangold', 'Joseph Kosinski', 'Sean Baker', 'Paul Feig', 'Steven Quale', 'Joachim Trier', 'Christopher Landon', 'Francis Lawrence', 'Matt Shakman', 'Coralie Fargeat', 'Frank Darabont', 'Christian Zübert', 'Robert Eggers', 'Julius Onah', 'Halina Reijn', 'Scott Derrickson', 'Scott Beck', 'Celine Song', 'Alex Scharfman', 'Harry Lighton', 'Raj Kumar Gupta', 'Kleber Mendonça Filho', 'Jared Bush', 'Julia Ducournau', 'Jafar Panahi', 'James Hawes', 'Christopher Nolan', 'Shivam Nair', 'Francis Ford Coppola', 'Julia Max', 'Eli Craig', 'Osgood Perkins', 'Sean Anders', 'Tim Burton', 'Denis Villeneuve', 'Darren Aronofsky', 'Felipe Vargas', 'Alan Alda', 'Fleur Fortune', 'George Lucas', 'Anthony Russo', 'James Watkins', 'Sean Byrne', 'Jon M. Chu', 'Mike Flanagan', 'J.C. Chandor', 'Rob Reiner', 'Spike Lee', 'Lee Isaac Chung', 'Leigh Janiak', 'Gaspar Noé', 'Elisabeth Röhm', 'Robert Zemeckis', 'Walter Salles', 'Quentin Tarantino', 'Chris Columbus', 'Kelly Marcel', 'Josh Ruben', 'Peter Jackson', 'Gia Coppola', 'Jonathan Eusebio', 'Gary Ross', 'Chad Stahelski', 'Justin Kurzel', 'Shawn Levy', 'Paul Weitz', 'Tim Mielants', 'Ethan Coen', 'David Fincher', 'Karan Sharma', 'Martin Scorsese', 'Adam Brooks', 'Joe Wright', 'M. Night Shyamalan', 'Alex Parkinson', 'Zoë Kravitz', 'Catherine Hardwicke', 'Jesse Armstrong', 'Jonathan Demme', 'Mel Gibson', 'Michael Shanks', 'Flying Lotus', 'Brad Furman', 'Jesse Eisenberg', 'Lana Wachowski', 'Barry Jenkins', 'Dougal Wilson', 'James Madigan', 'Richard Donner', 'Glenn Ficarra', 'Juan Carlos Fresnadillo', 'Steven Spielberg', 'David G. Derrick Jr.', 'James Cameron', 'Ilya Naishuller', 'Richard Linklater', 'Damien Chazelle', 'Sam Taylor-Johnson', 'Gene Stupnitsky', 'Emílio Domingos', 'Martin Campbell', 'Oliver Hermanus', 'Scarlett Johansson', 'Gints Zilbalodis', 'Fede Alvarez', 'Ben Stiller', 'Christian Gudegast', 'Joel Souza', 'Will Gluck', 'Robert Luketic', 'Barry Levinson', 'Kristen Stewart', 'Jillian Bell', 'Prince Dhiman', 'Timo Tjahjanto', 'Michael Mann', 'Kathryn Bigelow', 'Barbara Bialowas', 'M.J. Bassett', 'Yorgos Lanthimos', 'Colin Trevorrow', 'Matt Wilcox', 'Ti West', 'Sidney Lumet', 'Brian Helgeland', 'David Yarovesky', 'Lars von Trier', 'Tony Scott', 'Matt Reeves', 'Ron Howard', 'John Crowley', 'Akiva Schaffer', 'Irvin Kershner']
director = st.selectbox("🎬 감독 선택", director_list)

# 배우 목록
total=[]
actor_list=['Liz Carr', 'Zarrin Darnell-Martin', 'Jason Statham', 'Lance LeGault', 'Joseph Lee', 'Louis Minnaar', 'Jóhannes Haukur Jóhannesson', 'Abhishek Chauhan', 'Maria Tepavicharova', 'Carmen Hayward', 'Michael Wardle', 'Kumail Nanjiani', 'Jennifer Aniston', 'Greta Schröder', 'Monica Bellucci', 'Paris Vaughan', 'Dustin Hoffman', 'Shia LaBeouf', 'Stephen Kunken', 'Luis Guzmán', 'Martin Landau', 'Hristo Shopov', 'Franz Hartwig', 'Toshi Toda', 'T.J. Miller', 'Roberta Maxwell', 'Jack Noseworthy', 'J.D. Cannon', 'Ryan Gosling', 'Liam Cunningham', 'William Hurt', 'Michael Angelo Covino', 'Caoilfhionn Dunne', 'Andy Devine', 'Ludacris', 'Henry Cavill', 'Gerald McRaney', 'John Cho', 'Rafe Spall', 'Gina Gershon', 'Jim Cummings', 'David Proval', 'Tom Hanks', 'Patrick Van Horn', 'Tuva Novotny', 'Shenae Grimes-Beech', 'Pamela Anderson', 'Ben Whishaw', 'Stefania Gadda', 'Boden Johnston', 'Marlene Dietrich', 'Joju George', 'Paul Ben-Victor', 'Matthew Porretta', 'Issa Rae', 'Don Cheadle', 'Dwayne Johnson', 'Natasia Demetriou', 'Pom Klementieff', 'Jeanne Tripplehorn', 'Viola Davis', 'Damon Wayans Jr.', 'Andy Dick', 'Aries Spears', 'Tom Berenger', 'Edward Corbett', 'James Badge Dale', 'William Allen Young', 'Pratap Verma', 'Nicholas Braun', 'Scott Wilson', 'Gary Owens', 'Richard Bohringer', 'Olwen Catherine Kelly', 'Mel Gibson', " Peter O'Farrell", 'Naomi Grossman', 'Denise Richards', 'Otis Winston', 'Kirstie Alley', 'Choi Min-sik', 'Tiger Haynes', 'Cynthia Addai-Robinson', 'August Diehl', 'Konstantin Lavronenko', 'Nick Swardson', 'Barbet Schroeder', 'Lucy Russell', 'Brian Bedford', 'Jürgen Prochnow', 'Lily Rabe', 'Steven Elliot', 'Charles Walker', 'Claude Chagrin', 'Hana Toyoshima', 'Sam Trammell', 'Kenneth Nkosi', 'Paul Schneider', 'Hanna Huffman', 'Mia Goth', 'Alexander Calvert', 'Byron Capers', 'Zoe Saldaña', 'Blake Jenner', 'Tullio Carminati', 'Gerard Butler', 'Kayvan Novak', 'David Lengel', 'Jason Bateman', 'Gregory Ambrose Calderone', 'Crispin Glover', 'Eddie Jemison', 'Tao Zhao', 'Burt Reynolds', 'Robert Bathurst', 'Jim Caviezel', 'Sonoya Mizuno', 'Paul Rudd', 'Quincy Tyler Bernstine', 'William Fichtner', 'Navid Negahban', 'Robert Englund', 'Ye Soo-jung', " Michael O'Hearn", 'Helen Hunt', 'Zhi Wang', 'Paula Patton', 'Shila Ommi', 'David Ramsey', 'Jonathan Livingstone', 'Neil Bishop', 'James Logan', 'Lynda Boyd', 'Bill Greene', 'Richard Lage', 'Taylor Rice', 'Martin Wielgus', 'Jane Curtin', 'Andrew Bachelor', 'Carmen Electra', 'Uma Thurman', 'Niall Hayes', 'Laurent Labasse', 'Park Myung-shin', 'Ed Harris', 'Lee Jun-hyuk', 'Siddharth Dhananjay', 'Moran Atias', 'Nate Heller', 'Christian De Sica', 'Tiny Mills', 'Robert Downey Jr.', 'Jane Sibbett', 'Amandla Stenberg', 'Mila Lieu', 'Leonard Mudie', 'Mary Elizabeth Winstead', 'Jude Ciccolella', 'Strother Martin', 'Reginald Ballard', 'Alexander Settineri', 'Ben Lawson', 'Summer Glau', 'Nikolaj Coster-Waldau', 'Jack Lemmon', 'Stephen Root', 'Todd Louiso', 'Meryl Streep', 'Mckenna Grace', 'Roberts Blossom', 'Bill Hickman', 'Natale Bosco', 'Javier Bardem', 'Olivia Sanabia', 'Yoson An', 'Megan Liu', 'Tiarnie Coupland', 'Joe Pantoliano', 'Clarke Peters', 'Alexander Skarsgård', 'Alice Heffernan-Sneed', 'Kris Lemche', 'Harrison Ford', 'Johnny Knoxville', 'Patricia Hastie', 'Diong-Kéba Tacu', 'Enrique Murciano', 'Stephen Hunter', 'Joe Morton', 'Lisa Bonet', 'James Duval', 'Lena Endre', 'Chloë Sevigny', 'Michael Miccoli', 'Darlanne Fluegel', 'Dennis Alexio', 'Christine Belford', 'Peter McDonald', 'Poon Mitpakdee', 'Lance Henriksen', 'Shane McRae', 'Whitmer Thomas', 'Archana Rajan', 'Gary Riley', 'Harrison Page', 'Kim Yun-Seo', 'Jim Broadbent', 'Eric Keenleyside', 'David Paetkau', 'Cailee Spaeny', 'Raimo Rendi', 'Sean Maher', 'Mario Van Peebles', 'Christopher Collet', 'Karl Markovics', 'Brittany Hingle', 'Chuxiao Qu', 'Saoirse Ronan', 'Evangeline Rose', 'Jayne Brook', 'Robert De Niro', 'Jim Ishida', 'Christian Middelthon', 'Michael McShane', 'Meggan Lennon', 'Nuri Hazzard', " Conner O'Malley", 'Geraldine Viswanathan', 'Susan Blommaert', 'Gypsy Wood', 'Robert Wilfort', 'Nicholas Galitzine', 'Octavia Spencer', 'Tobe Nwigwe', 'Jeffrey Wright', 'Jordi Mollà', 'Lee Yeong-seok', " Bobb'e J. Thompson", 'Hiroyuki Sanada', 'Teri Polo', 'Stewart Finlay-McLennan', 'Ido Mosseri', 'Demi Moore', 'Archie Barnes', 'Sarita Joshi', 'Scotty Tovar', 'Sylvia Minassian', 'Karin Neuhäuser', 'Rodney Eastman', 'Jewel Staite', 'Apinya Sakuljaroensuk', 'Jennifer Lawrence', 'Frank Medrano', 'Emma Tremblay', " Matt O'Leary", 'Wim Opbrouck', 'Ally Sheedy', 'Solal Lucas', 'Linda Moran', 'Margaret Colin', 'Budd Friedman', 'Rob Mills', 'Rachel Sennott', 'Austin P. McKenzie', 'Walton Goggins', 'Diedrich Bader', 'Don Simpson', 'Gillian Anderson', 'Peter Bowles', 'Jeremy Renner', 'James Lancaster', 'Sung Kang', " Weeratham 'Norman'Wichairaksakui", 'Eric Dane', 'Tom Amandes', 'Eugene Khumbanyiwa', 'Agathe Rousselle', 'Raye Birk', 'Daniel Craig', 'Jason Mantzoukas', 'Ray Milland', 'Parker Posey', 'Jessica Sutta', 'William Ostrander', 'Tom Hiddleston', 'Kim Dickens', 'Michael C. Hall', 'Penny Santon', 'Lilly Krug', 'Kathryn Morris', 'Richard Jordan', 'Maria Latour', 'Marissa Ribisi', 'George Clooney', 'James Gandolfini', 'Raul Julia', 'Julio Oscar Mechoso', 'Killian Scott', 'Ty Burrell', 'Denzel Washington', 'Martin Balsam', 'Douglas Booth', 'Natalie Portman', 'Janel Parrish', 'Leslie Mann', 'Eve Ridley', 'Judy Reyes', 'Michael McDonald', 'Chika Ikogwe', 'Bostin Christopher', 'Lola Noh', 'Michael J. Fox', 'Seth Rogen', 'Jane Horrocks', 'Dan Marino', 'Zach Galifianakis', 'Bruce Davison', 'Jeff Branson', 'Nicholas Campbell', 'Elisabeth Locas', 'Mairead Devlin', 'Joe Viterelli', 'Jean Reno', 'Richard Dysart', 'Gregg Henry', 'Jay Villiers', 'Alan King', 'Steve Wall', 'Adam Baldwin', 'Michael Pitt', 'Dane Rhodes', 'John Candy', 'Julianne Hough', 'Dolph Lundgren', 'Sharlto Copley', 'Paul Anka', 'Victor Chatchawit Techarukpong', 'Javon Frazer', 'Kelly Van der Burg', 'Chip Hormess', 'Tom Cruise', 'Danny Glover', 'Eszter Balint', 'Tim Heidecker', 'Tania Saulnier', 'George Macready', 'Juliette Binoche', 'Geeta Agrawal Sharma', 'Rylan Jackson', 'Sean Hayes', 'Lincoln Pearson', 'Robert Downey Jr.', 'Hans Christian Blech', 'Paul Butler', 'Hayden Christensen', 'Craig Roberts', 'Michael Peña', 'Paul DeAngelo', 'Richmond Arquette', 'Mike Myers', 'Seth Gilliam', 'Daniel Rigamer', 'Benjamin Weir', 'Samantha Coughlan', 'Cathleen Nesbitt', 'Rick Worthy', 'Diane Venora', 'Olga Abrego', 'Mitchell LaFortune', 'Takato Yonemoto', 'Anne Serra', 'Ray Stevenson', 'Loris Diran', 'Tadanobu Asano', 'Waris Ahluwalia', 'George Leigh', 'Natasha Lyonne', 'Alexander Skarsgård', 'Michael B. Jordan', 'Richard Lawson', 'Demi Moore', 'Michael Kendall Kaplan', 'Chris Walley', 'Tilda Swinton', 'Matt Adler', 'Eric Davis', 'Liv Ullmann', 'Susan Ursitti', 'Bo Brinkman', 'Radha Mitchell', 'Robert Prosky', 'Sergi López', 'George Kennedy', 'Jake Ryan', 'John Ford Noonan', 'Emilija Baranac', 'Harry Dean Stanton', 'Allyn Rachel', 'Dana Ivey', 'Kevin James', 'Art Chudabala', 'Debra Feuer', 'Ann Dowd', 'Ian Edwards', 'Tom Rhys Harries', 'Andrew McCarthy', 'Diego Arnary', 'Max Martini', 'Jean-Claude Van Damme', 'Bronagh Gallagher', 'Sean Penn', 'Tim Preece', 'Ronobir Lahiri', 'David Horton', 'Ma Dong-seok', 'Tamara Tunie', 'Vince Vaughn', 'Alexander Arnold', 'Liam Neeson', 'Neal McDonough', 'Eric Johnson', 'Edward Barbanell', 'Sean Cory', 'Rod McLachlan', 'Kerry Walker', 'Noah Hathaway', 'James Brolin', 'Myriem Akheddiou', 'Tracey Walter', 'Stefan Kapicic', 'Maggie Q', 'Armie Hammer', 'Sylvester Stallone']
actors = st.multiselect("🎭 주요 배우 선택", actor_list)
actors_str = ", ".join(actors)

# 기타 정보
genre = st.selectbox("🎭 장르", list(genre_keywords.keys()))
duration = st.number_input("⏱ 상영 시간 (분)", 60, 240, 132)
country_list=['United States','United Kingdom','France','Canadas','Germany','Japan','India','Italy','Austrailia','China','Korea']
language_list=['English','EnglishSpanish','EnglishFrench','Japanese','EnglishGerman','Hindi','EnglishItalian','EnglishRussian','Spanish','Korean']
country = st.selectbox("🌍 국가", country_list)
language = st.selectbox("🗣 언어", language_list)
awarded = st.checkbox("🏆 수상 여부", value=True)
plot = st.text_area("📝 줄거리 요약", "In a futuristic world, dreams are used as currency...")

# 리뷰 생성 버튼
if st.button("📜 리뷰 생성하기"):
    keywords = genre_keywords.get(genre)
    selected_keywords = random.sample(keywords, k=min(3, len(keywords)))

    awards = "Yes" if awarded else "No"
    tone = "highly positive" if rating >= 7.0 else "neutral or critical"

    input_text = (
        f"Write a {tone} movie review in about 5 sentences using the keywords: "
        f"{', '.join(selected_keywords)}. "
        f"Plot: {plot}"
    )

    with st.spinner("리뷰 생성 중..."):
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = model.generate(
            input_ids,
            max_length=300,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=1
        )
        review = tokenizer.decode(outputs[0], skip_special_tokens=True)

    highlighted_review = highlight_keywords(review, selected_keywords)
    st.subheader("📝 생성된 리뷰")
    st.markdown(highlighted_review)
