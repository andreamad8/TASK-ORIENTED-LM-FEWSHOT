from collections import defaultdict
import pprint
import random
pp = pprint.PrettyPrinter(indent=2)

y1_set = ["O", "B", "I"]
y2_set = ['O', 'B-playlist', 'I-playlist', 'B-music_item', 'I-music_item', 'B-geographic_poi', 'I-geographic_poi', 'B-facility', 'I-facility', 'B-movie_name', 'I-movie_name', 'B-location_name', 'I-location_name', 'B-restaurant_name', 'I-restaurant_name', 'B-track', 'I-track', 'B-restaurant_type', 'I-restaurant_type', 'B-object_part_of_series_type', 'I-object_part_of_series_type', 'B-country', 'I-country', 'B-service', 'I-service', 'B-poi', 'I-poi', 'B-party_size_description', 'I-party_size_description', 'B-served_dish', 'I-served_dish', 'B-genre',  'I-genre', 'B-current_location', 'I-current_location', 'B-object_select', 'I-object_select', 'B-album', 'I-album', 'B-object_name', 'I-object_name', 'B-state', 'I-state', 'B-sort', 'I-sort', 'B-object_location_type', 'I-object_location_type', 'B-movie_type', 'I-movie_type', 'B-spatial_relation', 'I-spatial_relation', 'B-artist', 'I-artist', 'B-cuisine', 'I-cuisine', 'B-entity_name', 'I-entity_name', 'B-object_type', 'I-object_type', 'B-playlist_owner', 'I-playlist_owner', 'B-timeRange', 'I-timeRange', 'B-city', 'I-city', 'B-rating_value', 'B-best_rating', 'B-rating_unit', 'B-year', 'B-party_size_number', 'B-condition_description', 'B-condition_temperature']
domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
slot_by_domain = {"AddToPlaylist":['entity_name', 'playlist', 'artist', 'playlist_owner', 'music_item'],
                  "BookRestaurant":['party_size_number', 'state', 'restaurant_type', 'timeRange', 'city', 'party_size_description', 'sort', 'served_dish', 'spatial_relation', 'poi', 'restaurant_name', 'country', 'cuisine', 'facility'],
                  "GetWeather":['city', 'state', 'timeRange', 'geographic_poi', 'spatial_relation', 'current_location', 'country', 'condition_temperature', 'condition_description'],
                  "PlayMusic":['artist', 'album', 'service', 'music_item', 'track', 'year', 'sort', 'playlist', 'genre'],
                  "RateBook":['object_select', 'object_type', 'rating_value', 'best_rating', 'object_part_of_series_type', 'rating_unit', 'object_name'],
                  "SearchCreativeWork":['object_type', 'object_name'],
                  "SearchScreeningEvent":['movie_name', 'object_type', 'movie_type', 'spatial_relation', 'object_location_type', 'location_name', 'timeRange']}


def get_none_example(current_slot,dic_none,num):
    sent_for_none = dic_none[current_slot]
    random.shuffle(sent_for_none)
    return sent_for_none[:num]


def SNIPS(args,domain="AddToPlaylist",shots=1):

    with open(f".data/SNIPS/{domain}.txt", "r") as f:
        data_train = defaultdict(list)
        data_train_none = defaultdict(list)
        test = []
        for i, line in enumerate(f):
            line = line.strip()  # text \t label
            splits = line.split("\t")
            tokens = splits[0].split()
            l2_list = splits[1].split()

            dic = defaultdict(list)
            for t,bio in zip(tokens,l2_list):
                if(len(bio)==1):
                    #reset 
                    pass
                else:
                    dic[bio.replace("B-","").replace("I-","")].append(t)
            if(i<= 100):
                for k,v in dic.items():
                    data_train[k].append([splits[0]," ".join(v)])
                
                for s in slot_by_domain[domain]:
                    if(s not in dic.keys()):
                        data_train_none[s].append([splits[0],"none"])
            else:   
                test.append([" ".join(tokens),l2_list,dic])


    train = {}
    for k, v in data_train.items():
        train[k] = v[:int(shots*args.balanced)+1]
        train[k] += get_none_example(k,data_train_none,int(shots*(1-args.balanced))+1)
        random.shuffle(train[k])    
    return train, test[:501], list(train.keys()) # get last 500 example like coach


# for dom in ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]:
#     print(dom)
#     SNIPS(dom)
#     exit()

def get_example(split):
    lab = open(f".data/SNIPS/original_snips_data/{split}/label", "r") 
    seq = open(f".data/SNIPS/original_snips_data/{split}/seq.in", "r") 
    pairs = []
    for s,l in zip(seq,lab):
        pairs.append([s.replace("\n",""),l.replace("\n","").lower()])
    return pairs

def get_example_by_class(X):
    example_by_class = defaultdict(list)
    for idx, val in enumerate(X):
        example_by_class[val[1]].append([val[0],val[1]])
    return example_by_class

def get_false_example(dic,clas,num): 
    sent_false = []
    for k,v in dic.items():
        if(k!=clas):
            for sent in v:
                sent_false.append([sent[0],'false'])
    random.shuffle(sent_false)
    return sent_false[:num]

def SNIPS_get_intent(args):
    train, valid, test = get_example("train"),get_example("valid"),get_example("test")
    example_by_class = get_example_by_class(train)
    if(args.binary):
        train_shots = {}
        for k, v in example_by_class.items():
            train_shots[k] = [ [e[0],'true'] for e in v[:int(args.shots*args.balanced)+1]]
            train_shots[k] += get_false_example(example_by_class,k,int(args.shots*(1-args.balanced))+1)
            random.shuffle(train_shots[k])   
    else:
        train_shots = []
        for _, v in example_by_class.items(): 
            train_shots += v[:args.shots]
    return train_shots,valid,test
