import { Flex } from '@chakra-ui/core';
import React from 'react';
import { useQuery } from 'urql';

const Home = () => {
  const output = useQuery({
    query: `
      query { hello }
    `,
  });
  const res = output[0];

  if (res.fetching) return <p>Loading...</p>;
  if (res.error) return <p>Errored!</p>;

  return (
    <Flex padding="1.5rem">
      <h2>Vevo {res.data.hello}</h2>
    </Flex>
  );
};

export default Home;
