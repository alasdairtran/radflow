import { CSSReset, ThemeProvider } from '@chakra-ui/core';
import { cacheExchange } from '@urql/exchange-graphcache';
import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import { Client, dedupExchange, fetchExchange, Provider } from 'urql';
import Header from './components/Header';
import Home from './pages/Home';
import Vevo from './pages/Vevo';
import Wiki from './pages/Wiki';

const cache = cacheExchange({});

const client = new Client({
  url:
    process.env.NODE_ENV === 'production'
      ? 'https://api.radflow.ml/'
      : 'http://localhost:8205/',
  exchanges: [dedupExchange, cache, fetchExchange],
});

function App() {
  return (
    <Provider value={client}>
      <ThemeProvider>
        <CSSReset />
        <Router>
          <Header />
          <Switch>
            <Route exact path="/">
              <Home />
            </Route>
            <Route path="/vevo">
              <Vevo />
            </Route>
            <Route path="/wiki">
              <Wiki />
            </Route>
          </Switch>
        </Router>
      </ThemeProvider>
    </Provider>
  );
}

export default App;
