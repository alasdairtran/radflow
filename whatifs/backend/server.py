import pydgraph
from ariadne import ObjectType, QueryType, gql, make_executable_schema
from ariadne.asgi import GraphQL
from starlette.middleware.cors import CORSMiddleware

# Define types using Schema Definition Language (https://graphql.org/learn/schema/)
# Wrapping string in gql function provides validation and better error traceback
type_defs = gql("""
    type Query {
        people: [Person!]!
    }

    type Person {
        firstName: String
        lastName: String
        age: Int
        fullName: String
    }
""")

# Map resolver functions to Query fields using QueryType
query = QueryType()

# Resolvers are simple python functions


@query.field("people")
def resolve_people(*_):
    return [
        {"firstName": "John", "lastName": "Doe", "age": 21},
        {"firstName": "Bob", "lastName": "Boberson", "age": 24},
    ]


# Map resolver functions to custom type fields using ObjectType
person = ObjectType("Person")


@person.field("fullName")
def resolve_person_fullname(person, *_):
    return "%s %s" % (person["firstName"], person["lastName"])


# Create executable GraphQL schema
schema = make_executable_schema(type_defs, query, person)

# Create an ASGI app using the schema, running in debug mode
# app = GraphQL(schema, debug=True)
app = CORSMiddleware(GraphQL(schema, debug=True), allow_origins=[
                     'http://localhost:3200'], allow_methods=['POST'])
