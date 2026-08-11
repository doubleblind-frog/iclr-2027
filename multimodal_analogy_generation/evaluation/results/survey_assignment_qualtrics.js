Qualtrics.SurveyEngine.addOnReady(function () {

    var DISPENSER_URL = "https://script.google.com/macros/s/AKfycby6SOjBONwSR7QxwnqGDB1OVnslv4ajpmwKxdv8geDHD1oNFnsX6aCd2tyH0A7leTNK/exec";
    var that = this;

    that.disableNextButton();

    // Assignment lookup: group_id -> [pair_1_id, ..., pair_5_id]
    var ids = {
        1: ["076", "199", "026", "065", "113"],
        2: ["082", "102", "192", "122", "202"],
        3: ["095", "115", "199", "210", "159"],
        4: ["114", "110", "207", "051", "216"],
        5: ["100", "096", "015", "101", "211"],
        6: ["048", "027", "176", "152", "224"],
        7: ["129", "217", "034", "019", "146"],
        8: ["150", "190", "189", "048", "249"],
        9: ["077", "072", "032", "013", "014"],
        10: ["017", "220", "133", "088", "102"],
        11: ["218", "125", "098", "231", "068"],
        12: ["198", "065", "232", "037", "229"],
        13: ["131", "134", "111", "136", "224"],
        14: ["141", "164", "138", "116", "213"],
        15: ["101", "154", "067", "047", "072"],
        16: ["089", "238", "012", "039", "210"],
        17: ["035", "237", "050", "074", "081"],
        18: ["112", "033", "119", "231", "239"],
        19: ["125", "053", "203", "225", "171"],
        20: ["214", "060", "161", "045", "009"],
        21: ["173", "144", "017", "086", "109"],
        22: ["059", "240", "010", "239", "044"],
        23: ["088", "079", "246", "047", "219"],
        24: ["094", "177", "029", "117", "046"],
        25: ["120", "162", "153", "249", "170"],
        26: ["203", "056", "119", "237", "178"],
        27: ["033", "064", "130", "202", "081"],
        28: ["146", "110", "087", "070", "111"],
        29: ["013", "245", "206", "079", "201"],
        30: ["087", "242", "008", "039", "234"],
        31: ["250", "153", "090", "078", "162"],
        32: ["225", "235", "244", "174", "071"],
        33: ["109", "057", "035", "091", "228"],
        34: ["227", "076", "177", "195", "011"],
        35: ["131", "116", "129", "186", "058"],
        36: ["113", "212", "200", "169", "092"],
        37: ["155", "085", "248", "060", "097"],
        38: ["115", "056", "114", "142", "250"],
        39: ["180", "230", "094", "122", "120"],
        40: ["009", "058", "211", "121", "070"],
        41: ["107", "025", "074", "082", "045"],
        42: ["062", "130", "098", "071", "247"],
        43: ["097", "141", "028", "020", "226"],
        44: ["227", "147", "049", "064", "216"],
        45: ["049", "063", "099", "077", "054"],
        46: ["176", "221", "028", "173", "145"],
        47: ["091", "037", "233", "138", "068"],
        48: ["029", "086", "083", "139", "105"],
        49: ["218", "078", "016", "181", "207"],
        50: ["204", "034", "229", "190", "093"],
        51: ["158", "152", "112", "238", "161"],
        52: ["219", "104", "014", "026", "246"],
        53: ["134", "198", "031", "175", "220"],
        54: ["174", "194", "164", "158", "030"],
        55: ["046", "226", "132", "236", "151"],
        56: ["172", "217", "159", "095", "205"],
        57: ["140", "073", "133", "240", "213"],
        58: ["069", "154", "132", "092", "139"],
        59: ["093", "186", "144", "052", "194"],
        60: ["170", "215", "020", "096", "015"],
        61: ["209", "062", "204", "050", "140"],
        62: ["205", "172", "003", "171", "137"],
        63: ["206", "021", "057", "155", "235"],
        64: ["189", "228", "084", "104", "212"],
        65: ["150", "019", "008", "083", "059"],
        66: ["021", "208", "027", "007", "073"],
        67: ["147", "181", "055", "175", "160"],
        68: ["221", "010", "044", "232", "121"],
        69: ["234", "055", "007", "090", "195"],
        70: ["105", "145", "135", "169", "178"],
        71: ["124", "142", "053", "108", "031"],
        72: ["085", "236", "124", "244", "137"],
        73: ["209", "063", "136", "011", "245"],
        74: ["067", "180", "248", "201", "160"],
        75: ["214", "032", "016", "084", "100"],
        76: ["003", "208", "242", "025", "099"],
        77: ["052", "151", "018", "051", "200"],
        78: ["107", "230", "054", "215", "108"],
        79: ["069", "030", "233", "247", "117"],
        80: ["012", "018", "089", "135", "192"]
    };

    // Metaphor text lookup: group_id -> [pair_1_metaphor, ..., pair_5_metaphor]
    var metaphors = {
        1: ["Vodka freshly tapped from nature.", "Night is a toddler crawling on the floor.", "Coffee switches you on like a lightswitch.", "A healthy diet is preventive medicine for your organs.", "Listen to your uterus to recognise the signs of cancer."],
        2: ["A spoonful of NyQuil is a concentrated flock of sleep.", "My tongue is an old sock.", "My heart is a garden tired with autumn", "Failing to social distance is an explosion waiting to happen.", "The earth is a succulent Sunday roast."],
        3: ["Red bull is a battery.", "Vaseline can be used for art restoration.", "Night is a toddler crawling on the floor.", "The world is a paintbrush", "America is a gun."],
        4: ["Vaseline can smooth cracks in dried earth.", "Driving a Honda is sitting in a train on rails.", "The planet is a sinking ship.", "Global ice melting is as dangerous as a missile.", "Winter is a bony old crone"],
        5: ["Social media is a hamster wheel.", "Automotive safety is a gem to be treasured.", "Genetically modified food is a hidden predator on your plate.", "A social media feed is a roll of toilet paper.", "The world is a stage"],
        6: ["Strong hair gel has the grip of a lizard's feet.", "Colgate toothpaste throws a spotlight on your teeth.", "Her hair is a rippling, tossing sea", "An ancient anger exploded in his heart", "His fear is his prison."],
        7: ["Drink away your riches", "Work is a living hell", "Doritos can turn eating chips into a thrilling flavour adventure.", "Being addicted to smoking is being lost in a maze.", "It is sad to observe the fruits of ignorance"],
        8: ["We were in the jaws of death.", "My heart is a blank canvas waiting to be painted.", "My brain is a box of crayons", "Strong hair gel has the grip of a lizard's feet.", "He wanted to set sail on the ocean of love but instead wasted away in the desert."],
        9: ["Nescafé is a battery.", "The earth is a kitchen prep station for local ingredients.", "Krispy Kreme donuts can be found in Scottland, like the Loch Ness monster.", "Running is a calory crusher.", "Brämhults is a fresh carrot."],
        10: ["Risking change is like jumping into shark-ridden waters off a sinking ship.", "A friend is a treasure", "The burger is a flame of flavour", "Words can be deadly.", "My tongue is an old sock."],
        11: ["Your face is a canvas", "The canvas sails of a galleon ship are a swarm of vibrant butterflies.", "A work boot is a heavy-duty bulldozer.", "The stormy ocean was a raging bull.", "Hellman's mayonnaise turns meals into helium balloons."],
        12: ["My soul is a bird with broken wings.", "A healthy diet is preventive medicine for your organs.", "Airports are pools of money from passenger fees.", "SONY pushes music deep into your ears.", "The mind is a computer."],
        13: ["A party in your mouth", "The car can go as fast as the wind", "Ocean pollution is heading straight to the dinner table.", "The strawberries are as big as mountains", "His fear is his prison."],
        14: ["Love is in your bones", "Autumn is a depression", "Words can be deadly", "Industrial smokestacks are the surface proteins of a deadly virus.", "Time is a bus and I am running behind."],
        15: ["A social media feed is a roll of toilet paper.", "The crowd was a roaring river.", "The Powner drilll is Nelson Mandela breaking through Apartheid.", "Faber-Castell pencils are the true colors of nature.", "The earth is a kitchen prep station for local ingredients."],
        16: ["Your phone is a mouse trap.", "The company is letting money walk out the door.", "Cadbury chocolate is as creamy as fresh milk.", "Digital entertainment is a pallbearer for reading.", "The world is a paintbrush"],
        17: ["Dove cream is a flower.", "My future is a cloudy sky.", "The SUV's trunk is a dimensional fold that deletes physical length.", "Milo allows you to wake up as a champion.", "A blocked nose is a brick wall."],
        18: ["Faber-Castell pencils can be as brown as a dachshund.", "Yunky donut arrests your hunger.", "Prescription drug abuse is a daily schedule for mortality.", "The stormy ocean was a raging bull.", "Her mind was a lighthouse beacon to him."],
        19: ["The canvas sails of a galleon ship are a swarm of vibrant butterflies.", "A glass of beer is a good idea waiting to happen.", "The earth is a mother.", "India's culture is a salad bowl.", "Faith is a dangerous road."],
        20: ["Time is a thief", "Lego bricks can be a jetliner flying through the sky.", "An unexpected loss is a blow to the face.", "A sweet tooth is a predator chasing down your smile.", "The Nissan Juke is the Batmobil."],
        21: ["Friendship is a sheltering tree.", "Her pen was a knife", "Risking change is like jumping into shark-ridden waters off a sinking ship.", "Pepsi injects life right into your veins.", "An unauthorized car service is a predatory dinosaur waiting to strike."],
        22: ["Spicy ketchup is liquid fuel for a flame.", "He had a full bag of memories to unload.", "The music will punch you through the headphones.", "Her mind was a lighthouse beacon to him.", "Coca Cola is fuel for your thoughts."],
        23: ["Words can be deadly.", "Nivea can smooth even a prickly cactus.", "He liked the gym as much as a case of the measles.", "Faber-Castell pencils are the true colors of nature.", "Your mind is a powerful tool"],
        24: ["An open book is an umbrella shielding a child from a storm of smartphones.", "Her hair is like the curling mist", "If not cooled correctly, vegetables are dead corpses.", "A brain is not as tough to crack as a walnut", "Clear vision is the lens that turns art into reality."],
        25: ["Human hunters are the ultimate game in nature's flipped hierarchy.", "Anger is a demon that preys on the soul.", "My love is a summer day.", "He wanted to set sail on the ocean of love but instead wasted away in the desert.", "Failure is a treacherous pit."],
        26: ["The earth is a mother.", "An iron can smooth the dunes in the desert.", "Prescription drug abuse is a daily schedule for mortality.", "My future is a cloudy sky.", "Her love is a hurricane."],
        27: ["Yunky donut arrests your hunger.", "The lollipop is carved straight out of an apple.", "Global warming is a ticking bomb", "The earth is a succulent Sunday roast.", "A blocked nose is a brick wall."],
        28: ["It is sad to observe the fruits of ignorance", "Driving a Honda is sitting in a train on rails.", "Mugler perfume is a flower.", "Fries are as addictive as cigarettes.", "Ocean pollution is heading straight to the dinner table."],
        29: ["Running is a calory crusher.", "He is the sun of my sky.", "The ocean is a turbid grave.", "Nivea can smooth even a prickly cactus.", "The blazing sun is a ballroom dancer."],
        30: ["Mugler perfume is a flower.", "He is drowning in a sea of work.", "Be a tourist in Italy by eating Barilla pasta.", "Digital entertainment is a pallbearer for reading.", "I really hope society wakes up soon."],
        31: ["Her forehead was like an old map.", "My love is a summer day.", "Pizza delivery is a paper airplane flying through the city.", "Evian is a mountain in a bottle.", "Anger is a demon that preys on the soul."],
        32: ["India's culture is a salad bowl.", "My computer slipped into a coma.", "He is the shining star of our school.", "Hate is a blizzard", "Coffee beans are heavy eyelids opening wide."],
        33: ["An unauthorized car service is a predatory dinosaur waiting to strike.", "A two-wheel drive vehicle is a horse missing its hind legs.", "Dove cream is a flower.", "The penguin's last continent is a shopping bag.", "The classroom was a zoo."],
        34: ["Life is a maze.", "Vodka freshly tapped from nature.", "Her hair is like the curling mist", "My mouth is a music box.", "Resisting vaccines is like refusing to cross a professionally engineered bridge."],
        35: ["A party in your mouth", "Industrial smokestacks are the surface proteins of a deadly virus.", "Drink away your riches", "Love is a dead end.", "This ketchup is a spice bomb."],
        36: ["Listen to your uterus to recognise the signs of cancer.", "Thought is a vulture", "Our love is a board game.", "Every sin is a wage you must pay.", "Coca Cola is a power chord."],
        37: ["A heartbear is like a never-ending stampede of buffalo.", "Parship is a zipper bringing couples together.", "He should be in his car now, driving away, and yet here he was, like a fly caught in a spiderweb.", "Lego bricks can be a jetliner flying through the sky.", "A strip of adhesive tape is a flypaper trap catching airplanes in mid-air."],
        38: ["Vaseline can be used for art restoration.", "An iron can smooth the dunes in the desert.", "Vaseline can smooth cracks in dried earth.", "Adam did not understand the root of the crisis", "Her forehead was like an old map."],
        39: ["Her smile is a warm fire that melts my heart.", "The professor was a guiding light for him.", "An open book is an umbrella shielding a child from a storm of smartphones.", "Failing to social distance is an explosion waiting to happen.", "Human hunters are the ultimate game in nature's flipped hierarchy."],
        40: ["The Nissan Juke is the Batmobil.", "This ketchup is a spice bomb.", "The world is a stage", "Select the colour palette of your morning toast.", "Fries are as addictive as cigarettes."],
        41: ["You cannot afford to gamble with your dental health.", "A deadline is a trap closing in on a worker.", "Milo allows you to wake up as a champion.", "A spoonful of NyQuil is a concentrated flock of sleep.", "A sweet tooth is a predator chasing down your smile."],
        42: ["Your smile is a row of bright lightbulbs.", "Global warming is a ticking bomb", "A work boot is a heavy-duty bulldozer.", "Coffee beans are heavy eyelids opening wide.", "He looked neither one way nor the other way but sat like a carved image."],
        43: ["A strip of adhesive tape is a flypaper trap catching airplanes in mid-air.", "Love is in your bones", "Jaywalking is a death sentence.", "Buying cigarettes is throwing money away.", "Laughter is the best medicine."],
        44: ["Life is a maze.", "The faculty meeting was a nightmare", "Heinz ketchup is a stack of fresh tomatoes in a bottle.", "The lollipop is carved straight out of an apple.", "Winter is a bony old crone"],
        45: ["Heinz ketchup is a stack of fresh tomatoes in a bottle.", "Education is a factory that shapes individuals into uniform blocks.", "A high heel shoe is a striking snake.", "Nescafé is a battery.", "An ergonomic pillow is a sleep supplement."],
        46: ["Her hair is a rippling, tossing sea", "He was a cheetah in the race.", "Jaywalking is a death sentence.", "Friendship is a sheltering tree.", "I was alone in a sea of unknown faces"],
        47: ["The penguin's last continent is a shopping bag.", "SONY pushes music deep into your ears.", "The soul is envy's favourite meal.", "Words can be deadly", "Hellman's mayonnaise turns meals into helium balloons."],
        48: ["If not cooled correctly, vegetables are dead corpses.", "Pepsi injects life right into your veins.", "A bottle cap is an oyster shell hiding a treasure.", "Your heart is calling", "Tabasco sauce is a fire extinguisher."],
        49: ["Your face is a canvas", "Evian is a mountain in a bottle.", "Time is a highway.", "Her soul is a quiet sea.", "The planet is a sinking ship."],
        50: ["The mind is a glass fortress.", "Doritos can turn eating chips into a thrilling flavour adventure.", "The mind is a computer.", "My heart is a blank canvas waiting to be painted.", "Two cans of bug spray are a holy cross."],
        51: ["Alcohol is a friend.", "An ancient anger exploded in his heart", "Faber-Castell pencils can be as brown as a dachshund.", "The company is letting money walk out the door.", "An unexpected loss is a blow to the face."],
        52: ["Your mind is a powerful tool", "Floslek Sun Care is an oasis of shade on a scorching summer day.", "Brämhults is a fresh carrot.", "Coffee switches you on like a lightswitch.", "He liked the gym as much as a case of the measles."],
        53: ["The car can go as fast as the wind", "My soul is a bird with broken wings.", "Healthcare is a coin-operated service.", "Her hair is a bird's nest", "A friend is a treasure"],
        54: ["Hate is a blizzard", "My mind is a desert and I am an explorer.", "Autumn is a depression", "Alcohol is a friend.", "The pharmaceutical industry is a money machine."],
        55: ["Clear vision is the lens that turns art into reality.", "Laughter is the best medicine.", "Love is a hear-shaped donut", "The French bourgeoisie has rushed into a blind alley.", "The seed of life was planted on our planet"],
        56: ["Fear is a lock", "Work is a living hell", "America is a gun.", "Red bull is a battery.", "The moon's crescent is a smile."],
        57: ["Chocolate as smooth as silk", "The harvester has become the crop.", "The burger is a flame of flavour", "He had a full bag of memories to unload.", "Time is a bus and I am running behind."],
        58: ["McCafé coffee is fuel in a cup.", "The crowd was a roaring river.", "Love is a hear-shaped donut", "Coca Cola is a power chord.", "Your heart is calling"],
        59: ["Two cans of bug spray are a holy cross.", "Love is a dead end.", "Her pen was a knife", "A polar bear's body is a fracturing iceberg.", "My mind is a desert and I am an explorer."],
        60: ["Failure is a treacherous pit.", "True love is like plum blossoms in bloom.", "Buying cigarettes is throwing money away.", "Automotive safety is a gem to be treasured.", "Genetically modified food is a hidden predator on your plate."],
        61: ["The sun is a wounded deer.", "Your smile is a row of bright lightbulbs.", "The mind is a glass fortress.", "The SUV's trunk is a dimensional fold that deletes physical length.", "Chocolate as smooth as silk"],
        62: ["The moon's crescent is a smile.", "Fear is a lock", "Drinking vodka is absolute paradise.", "Faith is a dangerous road.", "Words burn"],
        63: ["The ocean is a turbid grave.", "Driving safety is a lion tamer.", "A two-wheel drive vehicle is a horse missing its hind legs.", "A heartbear is like a never-ending stampede of buffalo.", "My computer slipped into a coma."],
        64: ["My brain is a box of crayons", "The classroom was a zoo.", "The President spreads are a palette of flavours.", "Floslek Sun Care is an oasis of shade on a scorching summer day.", "Thought is a vulture"],
        65: ["We were in the jaws of death.", "Being addicted to smoking is being lost in a maze.", "Be a tourist in Italy by eating Barilla pasta.", "A bottle cap is an oyster shell hiding a treasure.", "Spicy ketchup is liquid fuel for a flame."],
        66: ["Driving safety is a lion tamer.", "The snow is a soft bed.", "Colgate toothpaste throws a spotlight on your teeth.", "Healthy eating is a sport.", "The harvester has become the crop."],
        67: ["The faculty meeting was a nightmare", "Her soul is a quiet sea.", "Fear makes monsters out of nothing.", "Her hair is a bird's nest", "An educated women is a danger."],
        68: ["He was a cheetah in the race.", "The music will punch you through the headphones.", "Coca Cola is fuel for your thoughts.", "Airports are pools of money from passenger fees.", "Select the colour palette of your morning toast."],
        69: ["I really hope society wakes up soon.", "Fear makes monsters out of nothing.", "Healthy eating is a sport.", "Pizza delivery is a paper airplane flying through the city.", "My mouth is a music box."],
        70: ["Tabasco sauce is a fire extinguisher.", "I was alone in a sea of unknown faces", "The sky is the limit", "Every sin is a wage you must pay.", "Her love is a hurricane."],
        71: ["Wildife is a living symphony orchestra.", "Adam did not understand the root of the crisis", "A glass of beer is a good idea waiting to happen.", "Tic tacs are as sweet and delicious as strawberries.", "Healthcare is a coin-operated service."],
        72: ["Parship is a zipper bringing couples together.", "The French bourgeoisie has rushed into a blind alley.", "Wildife is a living symphony orchestra.", "He is the shining star of our school.", "Words burn"],
        73: ["The sun is a wounded deer.", "Education is a factory that shapes individuals into uniform blocks.", "The strawberries are as big as mountains", "Resisting vaccines is like refusing to cross a professionally engineered bridge.", "He is the sun of my sky."],
        74: ["The Powner drilll is Nelson Mandela breaking through Apartheid.", "Her smile is a warm fire that melts my heart.", "He should be in his car now, driving away, and yet here he was, like a fly caught in a spiderweb.", "The blazing sun is a ballroom dancer.", "An educated women is a danger."],
        75: ["Time is a thief", "Krispy Kreme donuts can be found in Scottland, like the Loch Ness monster.", "Time is a highway.", "The President spreads are a palette of flavours.", "Social media is a hamster wheel."],
        76: ["Drinking vodka is absolute paradise.", "The snow is a soft bed.", "He is drowning in a sea of work.", "A deadline is a trap closing in on a worker.", "A high heel shoe is a striking snake."],
        77: ["A polar bear's body is a fracturing iceberg.", "The seed of life was planted on our planet", "The Publicis Singapore pencils wish you festive greetings.", "Global ice melting is as dangerous as a missile.", "Our love is a board game."],
        78: ["You cannot afford to gamble with your dental health.", "The professor was a guiding light for him.", "An ergonomic pillow is a sleep supplement.", "True love is like plum blossoms in bloom.", "Tic tacs are as sweet and delicious as strawberries."],
        79: ["McCafé coffee is fuel in a cup.", "The pharmaceutical industry is a money machine.", "The soul is envy's favourite meal.", "He looked neither one way nor the other way but sat like a carved image.", "A brain is not as tough to crack as a walnut"],
        80: ["Cadbury chocolate is as creamy as fresh milk.", "The Publicis Singapore pencils wish you festive greetings.", "Your phone is a mouse trap.", "The sky is the limit", "My heart is a garden tired with autumn"]
    };

    
    function writeFields(groupId) {
        var pairIds  = ids[groupId];
        var pairMets = metaphors[groupId];
        Qualtrics.SurveyEngine.setEmbeddedData("group_id",        String(groupId));
        Qualtrics.SurveyEngine.setEmbeddedData("pair_1_id",       pairIds[0]);
        Qualtrics.SurveyEngine.setEmbeddedData("pair_1_metaphor", pairMets[0]);
        Qualtrics.SurveyEngine.setEmbeddedData("pair_2_id",       pairIds[1]);
        Qualtrics.SurveyEngine.setEmbeddedData("pair_2_metaphor", pairMets[1]);
        Qualtrics.SurveyEngine.setEmbeddedData("pair_3_id",       pairIds[2]);
        Qualtrics.SurveyEngine.setEmbeddedData("pair_3_metaphor", pairMets[2]);
        Qualtrics.SurveyEngine.setEmbeddedData("pair_4_id",       pairIds[3]);
        Qualtrics.SurveyEngine.setEmbeddedData("pair_4_metaphor", pairMets[3]);
        Qualtrics.SurveyEngine.setEmbeddedData("pair_5_id",       pairIds[4]);
        Qualtrics.SurveyEngine.setEmbeddedData("pair_5_metaphor", pairMets[4]);
    }

    fetch(DISPENSER_URL, { redirect: "follow" })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            var groupId = data.groupId;
            if (!groupId) {
                Qualtrics.SurveyEngine.setEmbeddedData("group_id", "full");
            } else {
                writeFields(groupId);
            }
            that.enableNextButton();
        })
        .catch(function (err) {
            console.error("Dispenser failed:", err);
            writeFields(Math.ceil(Math.random() * 80));  // fallback
            that.enableNextButton();
        });

});