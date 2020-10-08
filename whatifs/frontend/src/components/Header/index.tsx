import { Box, Flex, Heading, Text } from '@chakra-ui/core';
import React from 'react';
import { Link } from 'react-router-dom';

const MenuItems = ({ children }: any) => (
  <Text mt={{ base: 4, md: 0 }} mr={6} display="block">
    {children}
  </Text>
);

const Header = (props: any) => {
  const [show, setShow] = React.useState(false);
  const handleToggle = () => setShow(!show);

  return (
    <Flex
      as="nav"
      align="center"
      justify="space-between"
      wrap="wrap"
      padding="1.5rem"
      bg="blue.500"
      color="white"
      {...props}
    >
      <Flex align="center" mr={5}>
        <Heading as="h1" size="lg">
          RADflow
        </Heading>
      </Flex>

      <Box display={{ sm: 'block', md: 'none' }} onClick={handleToggle}>
        <svg
          fill="white"
          width="12px"
          viewBox="0 0 20 20"
          xmlns="http://www.w3.org/2000/svg"
        >
          <title>Menu</title>
          <path d="M0 3h20v2H0V3zm0 6h20v2H0V9zm0 6h20v2H0v-2z" />
        </svg>
      </Box>

      <Box
        display={{ sm: show ? 'block' : 'none', md: 'flex' }}
        width={{ sm: 'full', md: 'auto' }}
        alignItems="center"
        flexGrow={1}
      >
        <Link to="/">
          <MenuItems>Home</MenuItems>
        </Link>
        <Link to="/wiki">
          <MenuItems>Wiki</MenuItems>
        </Link>
        <Link to="/vevo">
          <MenuItems>Vevo</MenuItems>
        </Link>
      </Box>
    </Flex>
  );
};

export default Header;
