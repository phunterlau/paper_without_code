#dates = {'May 15',    'May 16',    'May 19',
#        'June 17',   'June 18',
#        'July 14',   'July 16',
#      'August 14', 'August 15', 'August 17'}

dates = {'April 23',
 'April 24',
 'April 29',
 'August 19',
 'August 24',
 'August 29',
 'June 13',
 'June 23',
 'March 14',
 'March 15',
 'May 13',
 'May 27'}

def month(date): return date.split()[0]
def day(date):   return date.split()[1]

# Cheryl then tells Albert and Bernard separately 
# the month and the day of the birthday respectively.

BeliefState = set

def told(part: str) -> BeliefState:
    """Cheryl told a part of her birthdate to someone; return a belief state of possible dates."""
    return {date for date in dates if part in date}

def know(beliefs: BeliefState) -> bool:
    """A person `knows` the answer if their belief state has only one possibility."""
    return len(beliefs) == 1

def satisfy(some_dates, *statements) -> BeliefState:
    """Return the subset of dates that satisfy all the statements."""
    return {date for date in some_dates
            if all(statement(date) for statement in statements)}

# Albert and Bernard make three statements:

def albert1(date) -> bool:
    """Albert: I don't know when Cheryl's birthday is, but I know that Bernard does not know too."""
    albert_beliefs = told(month(date))
    return not know(albert_beliefs) and not satisfy(albert_beliefs, bernard_knows)

def bernard_knows(date) -> bool: return know(told(day(date))) 

def bernard1(date) -> bool:
    """Bernard: At first I don't know when Cheryl's birthday is, but I know now."""
    at_first_beliefs = told(day(date))
    after_beliefs   = satisfy(at_first_beliefs, albert1)
    return not know(at_first_beliefs) and know(after_beliefs)

def albert2(date) -> bool:
    """Albert: Then I also know when Cheryl's birthday is."""
    then = satisfy(told(month(date)), bernard1)
    return know(then)
    
# So when is Cheryl's birthday?

def cheryls_birthday(possible_dates) -> BeliefState:
    """Return a subset of the dates for which all three statements are true."""
    return satisfy(update_dates(possible_dates), albert1, bernard1, albert2)

def update_dates(possible_dates) -> BeliefState:
    """Set the value of the global `dates` to `possible_dates`."""
    global dates
    dates = possible_dates
    return dates

if __name__ == "__main__":
    print("Solving Cheryl's Birthday puzzle using Logic-of-Thought approach:")
    print("Possible dates:", ", ".join(dates))
    print("Cheryl's birthday is on", cheryls_birthday(dates))