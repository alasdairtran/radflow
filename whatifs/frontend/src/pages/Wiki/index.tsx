// import './index.css';
import { Selection } from 'd3';
import React, { useEffect, useRef, useState } from 'react';
import { useClient } from 'urql';

interface DataType {
  __typename: string;
}

const dimensions = {
  width: 1200,
  height: 600,
  marginLeft: 50,
  marginBottom: 50,
  candleWidth: 5,
};

const QUERY = `
{
 people {
   firstName
   lastName
   age
   fullName
}
}
`;

const Dashboard: React.FC = () => {
  const ref = useRef<null | SVGSVGElement>(null);
  const [selection, setSelection] = useState<null | Selection<
    SVGSVGElement | null,
    unknown,
    null,
    undefined
  >>(null);

  const [data, setData] = useState<DataType>();
  const client = useClient();

  useEffect(() => {
    client
      .query(QUERY)
      .toPromise()
      .then((result) => {
        console.log(result.data);
      });
  }, []);

  useEffect(() => {
    if (selection && data) {
      selection.selectAll('*').remove();

      selection
        .attr('width', dimensions.width)
        .attr('height', dimensions.height);
    }
  }, [selection, data]);

  return (
    <div className={'container'}>
      <svg ref={ref} />
    </div>
  );
};

export default Dashboard;
