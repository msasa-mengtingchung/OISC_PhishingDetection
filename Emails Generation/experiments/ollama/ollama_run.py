from ollama import chat
import time
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0))

start = time.time()
                # (iii) Prize or Lottery Winning
                # (iv) Fake Invoice or Payment Request
                # (v) Charity Scam
                # (vi) Account Security
                # (vii) Tax Refund Scam
                # (viii) Job Offer
                # (ix) Social Media Notification
                # (x) COVID-19 Related Scam
                # (xi) Law Breaking Activity
                # (xii) War-Related Aid

f = open("generatedText.txt", "w")

for x in range(2):

    stream = chat(
        model='llama3.2',
        messages=[{'role': 'user',
                'content': '''I am builing phishing detection model. Please generate me a phishing email to test the model. 
                    (i) Urgent Account Verification
                    (ii) Financial Transaction

                    The generated email about one of the random topics above should be convincing enough to trick the user into clicking on the link 
                    and use fake American real names as sender names or mention the company name.
                    The link should be a fake link, do not use real links but looks like a real link.
                    Do not mention the name of recipient because you do not know it, use generic terms like "Dear Customer" or others decent way to call recipient. 
                    Use thiese common words to geenrate the malicious and convincing content. The frequency of terms should be as follows:
                    company	statements	information	email	within	please	professional	report	adobe	price	securities	investment	money	time	future	security	news	business	stock	next	forward	number	software	advice	products	including	account	companies	also	bank	office	looking	made	could	stocks	service	contact	market	million	energy	part	best	shares	many	need	services	events	home	make	international	section	address	companys	message	wysak	performance	fact	based	provided	stop	expectations	interest	would	send	like	current	mail	results	lottery	newsletter	world	program	cialis	must	regards	meaning	know	wish	risks	want	dollars	prices	states	funds	order	investors	differ	name	terms	internet	days	claim	trade	registered	thanks	country	transaction	year	basin	numbers	take	fund	investing	windows	photoshop	private	since	position	uncertainties	without	certain	microsoft	customers	shall	projections	first	material	party	online	united	target	cause	offer	today	provide	even	look	estimates	believe	paid	full	macromedia	years	free	prescription	well	watch	voip	technology	product	corporation	industry	read	come	featured	reply	include	revenue	change	thousand	agent	system	acquisition	sell	express	stem	wyoming	exchange	projects	dont	assumptions	geec	release	visit	sent	remember	lauraan	major	readers	subscribers	limited	save	factors	going	offers	list	winning	site	transfer	phone	government	viagra	action	last	long	feel	cash	river	good	agreement	special	vcsc	note	reserves	might	involve	solutions	claims	ooking	prize	management	seek	inherent	examples	announced	speculative	publisher	brand	ever	family	growth	avoid	process	foreign	details	hours	receive	dear	hello	respect	words	computer	kind	place	help	soon	worldwide	research	find	third	necessary	used	subject	past	soft	give	expect	great	times	inform	plans	always	total	occur	prior	notice	thank	high	back	share	risk	letter	south	life	anticipated	names	contains	hundred	work	properties	huge	keep	every	access	acquire	possible	winners	assistance	mailings	range	reference	receiving	marketing	month	pease	said	award	assets	entertainment	team	manager	press	project	participants	potential	publication	among	sincerely	identified	forwardlooking	suite	state	quality	expects	financial	promotional	africa	provider	america	president	vocalscape	tabs	congratulations	believes	aware	systems	coud	less	newsetter	people	wide	technoogy	three	category	voice	selected	little	upon	months	green	batch	file	illustrator	shipping	cost	objectives	conflict	available	flash	development	herein	natural	confidential	therefore	presently	late	known	representative	petroleum	much	discussions	wysk	five	source	nothing	approved	studio	mentioned	deposit	cheaper	acceptance	data	week	anticipates	corel	acrobat	intend	none	sold	assist	others	indicating	real	represent	predictions	customer	continue	point	resuts	area	intent	advised	actual	regarding	public	safe	national	network	attention	needed	expected	programs	weeks	pills	value	popular	power	pertaining	form	required	show	milion	bring	feet	drugs	shoud	foresee	emai	mobile	given	dreamweaver	fireworks	understood	membership	immediately	80cs	group	advises	deciding	construed	trust	held	recent	view	later	waste	another	personal	currently	region	announces	cubic	actions	already	materially	unsubscribe	never	director	retail	opinion	china	eogi	lucky	credit	advertisement	production	right	department	erections	breaking	play	invest	incude	highly	wind	additional	website	promotions	notes	emerging	support	premiere	check	materia	advisor	solicitation	petroeum	enable	technologies	able	around	store	proved	lose	main	entire	ability	members	double	hope	sources	powder	vinoble	understands	taken	various	term	payment	second	vocalscapes	draw	pack	yahoocom	financing	produced	complete	easy	mind	move	needs	unique	europe	effects	click	line	profile	symbol	goals	near	throughout	review	general	person	filings	rate	link	perfect	opportunity	match	placed	partner	emerson	producing	response	investor	event	positive	decided	nigeria	following	imageready	higher	officer	opportunities	experience	legal	statement	broker	attached	addresses	amount	date	massive	purposes	aggressive	communications	according	pleased	todays	pans	potentia	acts	half	purchase	forth	markets	quick	working	completed	stated	considered	historical	ticket	logo	distribute	reform	present	deposited	operate	telephone	positioned	compliance	increase	estimated	alert	accuracy	pill	western	litigation	invove	guarantee	contents	result	still	operating	shareholder	north	mnei	directly	rocket	communication	contained	numerous	brec	strong	blood	better	enter	resulting	fast	matter	processing	body	creative	premium	similar	poised	hard	success	entered	homes	important	st0ck	beliefs	protocol	case	internationa	worth	holding	unclaimed	documents	size	optout	claiming	assurance	simply	methane	proven	create	client	info	projected	live	really	raise	proprietary	notification	reliable	friday	delivery	profit	ballot	card	timing	think	medical	death	european	affiliated	several	clients	love	smith	partnership	serial	drawn	write	technoogies	large	successful	discreet	includes	contract	microcap	dolars	quote	short	generic	interested	explode	direct	supply	application	maiings	georgia	paced	sites	equity	immediate	start	choice	trading	conditions	compensation	aerofoam	become	penny	found	canada	official	materialy	neither	staff	minutes	gains	pubication	revenues	strategy	advertising	drew	processed	died	deal	offering	pubisher	websites	acquisitions	proposal	pink	percent	different	health	pick	history	sports	reease	wrongfuly	accounts	awesome	urgent	drug	natura	meant	individual	attorney	ready	interactive	specuative	consequently	remitted	father	media	countries	identify	tadalafil	husband	making	facts	diligence	january	friend	treatment	released	ship	using	corporate	lower	expenses	together	billion	meet	finance	youre	miion	undue	asked	delays	confidentiality	receipt	earning	manufacturer	businesses	anything	competitors	incuding	emails	deveopment	opinions	sector	gathered	radar	netfones	everyone	thing	select	head	active	small	sysync	complications	begin	compensated	discovered	beneficiary	meds	milennium	travel	unnecessary	things	baitexcelledemca	mission	youve	outstanding	bmxg	huifeng	final	problem	concern	works	global	control	effort	lump	firms	accounting	screens	call	rfid	cautioned	agreements	design	releases	beieves	west	search	board	judge	rolex	hence	innovative	negotiations	firm	chinese	existing	treated	correspondence	reliance	pain	nationa	secured	expert	drive	bankruptcy	constitutes	competition	takes	cell	wels	happy	constitute	request	issue	makes	related	tongue	sheets	came	confirm	sexual	capital	lots	sales

                    The output should be in the form of a json file incluing subject and body.
                    The output should be in the form of a json file incluing subject, body and topic.
                    Only generate the email content, do not generate python code and explaination.
                '''}],
        stream=True
    )

 

    for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
            f.write(chunk['message']['content'])

f.close()

done = time.time()
elapsed = done - start
print(elapsed/60)