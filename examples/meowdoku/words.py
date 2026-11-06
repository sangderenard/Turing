"""Words for the Wordle board.

DICTIONARIES are the word lists a game can play from; the ACTIVE one is
both where the answer is drawn (any word of the chosen length in it -- no
separate answer tier) and what is accepted as a guess:

    builtin   the five-letter lists below (``ANSWERS`` + ``EXTRA``)
    official  the official Wordle list (every word the game accepts), five
              letters, downloaded once into ``build/meowdoku/dictionaries``
    complete  the ENABLE word list (~173k English words), downloaded once;
              its letter count is settable (``LENGTHS``)

A download that fails leaves the game on ``builtin`` and says so."""

ANSWERS = """
about above abuse actor acute admit adopt adult after again agent agree ahead alarm album alert alike alive allow alone along alter among anger angle angry apart apple apply arena argue arise armor array aside asset audio audit avoid awake award aware badly baker bases basic basis beach began begin being below bench billy birth black blade blame blank blast blend bless blind block blood board boost booth bound brain brand brave bread break breed brick brief bring broad broke brown build built buyer cabin cable candy carry catch cause chain chair chalk chaos charm chart chase cheap check cheek cheer chess chest chief child china chose civil claim class clean clear clerk click cliff climb clock close cloud coach coast could count court cover craft crash cream crime cross crowd crown cruel crush curve cycle daily dance dated dealt death debut delay delta dense depth devil diary dirty doubt dozen draft drain drama drank drawn dream dress dried drift drill drink drive drove dying eager early earth eight elbow elder elect elite empty enemy enjoy enter entry equal error event every exact exist extra faith false fancy fatal fault fever fiber field fifth fifty fight final first fixed flame flash fleet flesh float flood floor flour fluid focus force forge forth forty forum found frame frank fraud fresh front frost fruit fully funny ghost giant given glass globe glory glove goods grace grade grain grand grant grape graph grasp grass grave great green greet grief grill grind gross group grown guard guess guest guide guilt habit happy harsh heart heavy hedge hello hence hobby honey honor horse hotel house human humor hurry ideal image imply index inner input issue ivory jelly jewel joint judge juice karma knife knock known label labor large laser later laugh layer learn lease least leave legal lemon level light limit linen liver local logic loose lower loyal lucky lunch lying magic major maker maple march match maybe mayor meant medal media mercy merit metal meter might minor minus mixed model money month moral motor mount mouse mouth movie music naive naked nerve never newly night noble noise north novel nurse occur ocean offer often olive onion order other ought outer owner paint panel panic paper party pasta patch pause peace pearl penny phase phone photo piano piece pilot pinch pitch pizza place plain plane plant plate plaza plead point polar pound power press price pride prime print prior prize probe proof proud prove pulse punch pupil puppy purse queen query quest quick quiet quilt quite quota quote radar radio raise rally ranch range rapid ratio reach react ready realm rebel refer reign relax reply rider ridge rifle right rigid risky rival river robot rocky roman rough round route royal rugby ruler rural saint salad sauce scale scene scope score sense serve seven shade shaft shake shall shame shape share sharp sheep sheet shelf shell shift shine shirt shock shoot shore short shout shown sight silly since sixth sixty skill skirt slate sleep slice slide slope small smart smell smile smoke snake snowy solar solid solve sorry sound south space spare spark speak speed spell spend spent spice spike spine spite split spoke sport spray squad stack staff stage stain stair stake stamp stand stare start state steak steal steam steel steep steer stick stiff still sting stock stone stood store storm story stove strap straw strip stuck study stuff style sugar suite sunny super swear sweet swift swing sword table taken taste teach teeth tempo tenth thank theft their theme there these thick thief thing think third those three threw throw thumb tiger tight timer title toast today token topic total touch tough tower toxic trace track trade trail train trait trash treat trend trial tribe trick tried troop truck truly trunk trust truth twice twist uncle under union unite unity until upper upset urban usage usual valid value vapor venue verse video virus visit vital vivid vocal voice voter wagon waste watch water weary weigh weird whale wheat wheel where which while white whole whose wider woman women world worry worse worst worth would wound write wrong wrote yield young youth zebra
""".split()

EXTRA = """
aback abbey abide abode abort abyss ached acids acorn adage adder admin adore adorn aegis affix afoot agape agile aging aglow agony aisle alias align allay alley alloy aloft aloof alpha altar amass amaze amber amble amend amiss ample amuse angel anise ankle annex annoy antic anvil aorta aphid apron aptly arbor ardor argon aroma arrow arson ascot ashen askew assay atlas atoll attic augur aunty avail avert avian awful awoke axial axiom azure bacon badge bagel baggy balmy banal banjo barge baron basal basil baste batch bathe baton bawdy bayou beady beard beast beech beefy befit beget begun belch belie belle belly bento beret berry beset bevel bezel bible bicep biddy bigot bijou biome birch bison bitty blaze bleak bleat bleed bleep blimp bliss blitz bloat blond bloom blown bluff blunt blurb blurt blush boast bobby boney bongo bonus boozy borax bosom bossy botch bough boule bowel boxer brace braid brake brash brass brawl brawn brine brink brisk broil brook broom broth brunt brush brute buddy budge buggy bugle bulge bulky bully bunch bunny burly burnt burst bused bushy butch butte buxom bylaw cacao cache cacti caddy cadet cagey cairn camel cameo canal canny canoe canon caper caput carat cargo carol carve caste cater cedar cello chafe chaff champ chant chard chasm cheat chide chili chill chime chirp chock choir choke chord chore chuck chump chunk churn chute cider cigar cinch circa civic clack clad clamp clang clank clash clasp cleat cleft clerk cling clink cloak clone cloth clout clove clown cluck clued clump clung coast cobra cocoa colon color comet comfy comic comma conch condo conic copse coral corer corny couch cough coupe covet covey cower coyly crack cramp crane crank crass crate crave crawl craze crazy creak credo creed creek creep creme crepe crept cress crest crick cried crier crimp crisp croak crock crone crony crook croup crude crumb crypt cubic cumin curio curly curry curse curvy cutie cyber cynic dairy daisy dally dandy datum daunt dealt debar debit decal decay decor decoy decry defer deign deity delve demon demur denim depot derby deter detox deuce dicey digit dimly diner dingo dingy diode dirge disco ditch ditto ditty diver dizzy dodge dodgy dogma doing dolly donor donut dopey doubt dough dowdy dowel downy dowry drake drape drawl dread dregs droit droll drone drool droop dross drove drown druid drunk dryer dryly duchy dully dummy dumpy dunce dusky dusty dutch duvet dwarf dwell dwelt eagle easel eaten eater ebony eclat edict edify eerie egret eking elate elegy elfin elide elope elude email embed ember emcee ender endow enema ennui ensue envoy epoch epoxy equip erase erect erode essay ester ether ethic ethos etude evade evict evoke exalt excel exert exile expel extol exult eying fable facet faint fairy fanny farce fatty fauna favor feast fecal feign fella felon femme femur fence feral ferry fetal fetch fetid fetus fewer fiend fiery filer filet filly filmy filth finch finer firer fishy fizzy fjord flack flail flair flake flaky flank flare flask fleck flick flier fling flint flirt flock flora floss flout flown fluff fluke flume flung flunk flush flute flyer foamy focal foggy foist folio folly foray forge forgo forte foyer frail freak freed freer fresh friar fried frill frisk fritz frock frond frost froth frown froze fudge fugue fully fungi funky furor furry fussy fuzzy gaffe gaily gamer gamma gamut gassy gaudy gauge gaunt gauze gavel gawky gayer gayly gazer gecko geeky geese genie genre gipsy girly girth given giver glade gland glare glaze gleam glean glide glint gloat gloom gloss glyph gnash gnome godly going golem golly gonad goner goody gooey goofy goose gorge gouge gourd grail graft grasp gravy graze greed grime grimy gripe groan groin groom grope grout grove growl gruel gruff grunt guava guild guile guise gulch gully gumbo gummy guppy gusto gusty gypsy hairy halve handy hardy harem harpy haste hasty hatch hater haunt haute haven havoc hazel heady heard heath heave hefty heist helix heron hilly hinge hippo hippy hoard hobby hoist holly homer honey horde horny hound hovel hover howdy human humid humph humus hunch hunky hurry husky hussy hutch hydro hyena hymen hyper icily icing idiom idiot idler idyll igloo iliac imbue impel inane inbox incur inept inert infer ingot inlay inlet inter intro ionic irate irony islet itchy ivory jaunt jazzy jerky jetty jiffy joist joker jolly joust juicy jumbo jumpy junta junto juror kappa kayak kebab khaki kinky kiosk kitty knack knave knead kneed kneel knelt knoll koala krill
""".split()

ALL = frozenset(ANSWERS) | frozenset(EXTRA)


import threading
import urllib.request
from pathlib import Path

DICTIONARY_DIR = Path(__file__).resolve().parents[2] / "build" / "meowdoku" / "dictionaries"
SOURCES = {
    "official": "https://raw.githubusercontent.com/tabatkins/wordle-list/main/words",
    "complete": "https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt",
}
LABELS = {"builtin": "Built-in", "official": "Official Wordle", "complete": "Complete (ENABLE)"}
LENGTHS = {"builtin": (5,), "official": (5,), "complete": tuple(range(3, 11))}
_LOCK = threading.Lock()
_CACHE: dict[str, list[str]] = {}


def _load(name: str) -> list[str]:
    """Every word of dictionary ``name`` (lower-case letters only), fetched
    once into DICTIONARY_DIR and read from there afterwards."""
    if name == "builtin":
        return sorted(ALL)
    with _LOCK:
        if name in _CACHE:
            return _CACHE[name]
        path = DICTIONARY_DIR / f"{name}.txt"
        if not path.exists():
            DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(SOURCES[name], timeout=30) as response:
                data = response.read().decode("utf-8")
            tmp = path.with_suffix(".tmp")
            tmp.write_text(data, encoding="utf-8")
            tmp.replace(path)
        found = sorted({w.strip().lower() for w in path.read_text(encoding="utf-8").split() if w.strip().isalpha()})
        _CACHE[name] = found
        return found


def dictionary(name: str, length: int) -> tuple[str, int, list[str], str]:
    """(name, length, words, note): the active dictionary's words of
    ``length`` letters.  An unknown name or length is coerced to what that
    dictionary offers; a failed download falls back to ``builtin``."""
    if name not in LABELS:
        name = "builtin"
    if length not in LENGTHS[name]:
        length = 5 if 5 in LENGTHS[name] else LENGTHS[name][0]
    try:
        found = [w for w in _load(name) if len(w) == length]
        return name, length, found, ""
    except Exception as error:      # noqa: BLE001 (the network or the disk: play on, say why)
        return "builtin", 5, sorted(ALL), f"{LABELS[name]} unavailable ({error}); playing the built-in list"
